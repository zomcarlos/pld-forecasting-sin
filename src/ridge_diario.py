"""
Ridge Regression Diário v3 — Walk-Forward Validation × 4 Submercados.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from data_loader import SUBMERCADOS, get_aligned_data, build_features
from reproducibility import save_reproducibility_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PRED_DIR = OUTPUT_DIR / 'predictions'
PRED_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 30
N_SPLITS = 4

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def run_ridge_for_submercado(submercado: str):
    print(f"\n{'='*40}")
    print(f" Ridge Regression - Submercado: {submercado}")
    print(f"{'='*40}")

    df_raw = get_aligned_data(submercado)
    print(f"  Observações: {len(df_raw)} | Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()}")

    pld_original = df_raw["pld"].copy()
    df_raw["pld"] = np.log(df_raw["pld"])

    df_feat = build_features(df_raw)
    df_feat = df_feat.dropna()

    target_col = "pld"
    feature_cols = [c for c in df_feat.columns if c != target_col]

    X = df_feat[feature_cols]
    y = df_feat[target_col]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    fold_metrics = []
    last_coef = None
    
    all_dates = []
    all_y_true = []
    all_y_pred = []
    all_rmse = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Padronização essencial para Ridge (penalidade L2 depende da escala); fit apenas no treino evita leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # alpha=1.0 controla a intensidade da regularização L2 (encolhimento dos coeficientes)
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred_log = model.predict(X_test_scaled)

        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test.values)
        pd.DataFrame({
            "data": X_test.index.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred
        }).to_csv(PRED_DIR / f"ridge_{submercado}_fold{fold+1}.csv", sep=";", index=False)


        fold_mae = mean_absolute_error(y_true, y_pred)
        fold_rmse = root_mean_squared_error(y_true, y_pred)
        fold_mape = mape(y_true, y_pred)

        fold_metrics.append({"fold": fold + 1, "mae": fold_mae, "rmse": fold_rmse, "mape": fold_mape})
        print(f"  Fold {fold + 1} | MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f} | MAPE: {fold_mape:.2f}%")
        
        all_dates.extend(X_test.index)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_rmse.extend([fold_rmse] * len(y_test))

        # Magnitude indica força da feature; sinal indica direção do impacto no preço
        last_coef = pd.DataFrame({
            "feature": feature_cols,
            "coef_abs": np.abs(model.coef_),
            "coef": model.coef_
        }).sort_values("coef_abs", ascending=False)
        
    all_y_pred = np.array(all_y_pred)
    all_rmse = np.array(all_rmse)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_dates, all_y_true, label='Observado', color='black', alpha=0.7)
    ax.plot(all_dates, all_y_pred, label='Previsto', color='red')
    ax.fill_between(
        all_dates,
        np.maximum(0, all_y_pred - all_rmse),
        all_y_pred + all_rmse,
        color='red', alpha=0.2, label='Margem de Erro (± RMSE)'
    )
    ax.set_title(f'Ridge Regression - Previsão de PLD ({submercado})')
    ax.set_xlabel('Data')
    ax.set_ylabel('PLD (R$/MWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'ridge_forecast_{submercado}.png', dpi=150)
    plt.close(fig)

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_mape = np.mean([m["mape"] for m in fold_metrics])

    print(f"\n  MÉDIA CV (Walk-Forward {N_SPLITS}x{TEST_SIZE} dias)")
    print(f"  MAE:  {avg_mae:.2f}")
    print(f"  RMSE: {avg_rmse:.2f}")
    print(f"  MAPE: {avg_mape:.2f}%")

    if last_coef is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(last_coef))
        top = last_coef.head(top_n)
        # Cores indicam o sentido do impacto: verde eleva o preço e vermelho reduz
        colors = ['red' if c < 0 else 'green' for c in top["coef"].values]
        ax.barh(range(top_n), top["coef_abs"].values, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Magnitude do Coeficiente (Absoluto)")
        ax.set_title(f"Importância dos Coeficientes - Ridge ({submercado})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"ridge_importance_{submercado}.png", dpi=150)
        plt.close(fig)

    return {"submercado": submercado, "mae": avg_mae, "rmse": avg_rmse, "mape": avg_mape}

def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_ridge_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "ridge_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - RIDGE REGRESSION (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "TEST_SIZE": TEST_SIZE,
        "N_SPLITS": N_SPLITS
    }
    
    model_params = {
        "model_type": "Ridge Regression",
        "scaler": "StandardScaler",
        "ridge_hyperparameters": {
            "alpha": 1.0,
            "random_state": 42
        }
    }
    
    save_reproducibility_report(
        model_name="ridge_diario",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )

if __name__ == "__main__":
    main()
