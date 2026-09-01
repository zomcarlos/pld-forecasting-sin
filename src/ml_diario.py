"""
LightGBM Diário v3 — Walk-Forward Validation × 4 Submercados.

Exógenas: ENA (afluência), EAR (nível reservatório %), CARGA (demanda MWmed).
Mesmas exógenas utilizadas no modelo SARIMAX para comparação justa.

Referências:
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  NIPS 2017. https://papers.nips.cc/paper/6907
- Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art
  with a look into the future. IJF, 30(4), 1030-1044.
  DOI: 10.1016/j.ijforecast.2014.08.008
- Lago, J. et al. (2018). Forecasting spot electricity prices. Applied Energy, 221,
  386-405. DOI: 10.1016/j.apenergy.2018.02.069
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from data_loader import SUBMERCADOS, get_aligned_data, build_features
from reproducibility import save_reproducibility_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SIZE = 30
N_SPLITS = 4

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run_ml_for_submercado(submercado: str):
    print(f"\n{'='*40}")
    print(f" LightGBM - Submercado: {submercado}")
    print(f"{'='*40}")

    df_raw = get_aligned_data(submercado)
    print(f"  Observações: {len(df_raw)} | Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()}")

    # Transformação logarítmica do target
    # Aplicada antes do feature engineering para que lags e médias móveis também fiquem em escala log
    pld_original = df_raw["pld"].copy()
    df_raw["pld"] = np.log(df_raw["pld"])

    df_feat = build_features(df_raw)
    # Remove as primeiras ~30 linhas com NaN geradas pelo cálculo das janelas móveis
    df_feat = df_feat.dropna()

    target_col = "pld"
    feature_cols = [c for c in df_feat.columns if c != target_col]

    X = df_feat[feature_cols]
    y = df_feat[target_col]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "n_estimators": 500,     # Limite superior; calibrado dinamicamente via early stopping
        "learning_rate": 0.05,    # Taxa conservadora para evitar overfitting
        "num_leaves": 31,         # Valor padrão balanceado para complexidade das árvores
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    fold_metrics = []
    last_importance = None
    
    all_dates = []
    all_y_true = []
    all_y_pred = []
    all_rmse = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Valid set para early stopping
        # 10% final do treino reservado para monitorar a perda fora da amostra e evitar overfitting
        val_size = int(len(X_train) * 0.1)
        X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
        X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],  # Interrompe o treino se a perda na validação não cair por 30 rodadas
        )

        y_pred_log = model.predict(X_test)

        # Reverter log
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test.values)

        fold_mae = mean_absolute_error(y_true, y_pred)
        fold_rmse = root_mean_squared_error(y_true, y_pred)
        fold_mape = mape(y_true, y_pred)

        fold_metrics.append({"fold": fold + 1, "mae": fold_mae, "rmse": fold_rmse, "mape": fold_mape})
        print(f"  Fold {fold + 1} | MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f} | MAPE: {fold_mape:.2f}%")
        
        all_dates.extend(X_test.index)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_rmse.extend([fold_rmse] * len(y_test))

        # Importância calculada pelo número de vezes que cada feature foi usada nas divisões das árvores
        last_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
    all_y_pred = np.array(all_y_pred)
    all_rmse = np.array(all_rmse)
    
    # Gráfico de previsão vs real
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_dates, all_y_true, label='Observado', color='black', alpha=0.7)
    ax.plot(all_dates, all_y_pred, label='Previsto', color='green')
    # Faixa de incerteza (±RMSE): aproximação prática, já que LightGBM não produz intervalos de confiança nativos
    ax.fill_between(
        all_dates,
        np.maximum(0, all_y_pred - all_rmse),
        all_y_pred + all_rmse,
        color='green', alpha=0.2, label='Margem de Erro (± RMSE)'
    )
    ax.set_title(f'LightGBM - Previsão de PLD ({submercado})')
    ax.set_xlabel('Data')
    ax.set_ylabel('PLD (R$/MWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'ml_forecast_{submercado}.png', dpi=150)
    plt.close(fig)
    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_mape = np.mean([m["mape"] for m in fold_metrics])

    print(f"\n  MÉDIA CV (Walk-Forward {N_SPLITS}x{TEST_SIZE} dias)")
    print(f"  MAE:  {avg_mae:.2f}")
    print(f"  RMSE: {avg_rmse:.2f}")
    print(f"  MAPE: {avg_mape:.2f}%")

    # Feature importance do último fold
    if last_importance is not None:
        print(f"\n  Top 10 Features ({submercado}):")
        print(last_importance.head(10).to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(last_importance))
        top = last_importance.head(top_n)
        ax.barh(range(top_n), top["importance"].values, color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importância")
        ax.set_title(f"Feature Importance - LightGBM ({submercado})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"feature_importance_{submercado}.png", dpi=150)
        plt.close(fig)

    return {"submercado": submercado, "mae": avg_mae, "rmse": avg_rmse, "mape": avg_mape}


def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_ml_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "ml_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - LIGHTGBM (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "TEST_SIZE": TEST_SIZE,
        "N_SPLITS": N_SPLITS
    }
    
    model_params = {
        "model_type": "LightGBM Regressor",
        "lgb_hyperparameters": {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30
        }
    }
    
    save_reproducibility_report(
        model_name="ml_diario",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )


if __name__ == "__main__":
    main()
