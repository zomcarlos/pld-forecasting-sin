"""
Threshold Autoregressive (TAR) Diário v4 — Regime Switching Observável.

Este modelo divide a série em dois regimes (crise/estresse vs normalidade)
com base em uma variável de limiar observável (EAR - Energia Armazenada),
e ajusta um modelo linear (OLS regularizado / Ridge) para cada regime.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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

class ThresholdRegressor:
    """Modelo econométrico de Regime Switching Observável (Threshold Regression)."""
    def __init__(self, threshold_col="ear_pct", threshold_val=30.0, alpha=1.0):
        self.threshold_col = threshold_col
        self.threshold_val = threshold_val
        # Usamos Ridge (OLS regularizado) internamente para cada regime
        # para garantir estabilidade caso um dos regimes tenha poucas amostras.
        self.model_low = Ridge(alpha=alpha)
        self.model_high = Ridge(alpha=alpha)
        
    def fit(self, X, y):
        mask = X[self.threshold_col] < self.threshold_val
        if mask.sum() > 0:
            self.model_low.fit(X[mask], y[mask])
        if (~mask).sum() > 0:
            self.model_high.fit(X[~mask], y[~mask])
        return self
        
    def predict(self, X):
        pred = np.zeros(len(X))
        mask = X[self.threshold_col] < self.threshold_val
        if mask.sum() > 0:
            pred[mask] = self.model_low.predict(X[mask])
        if (~mask).sum() > 0:
            pred[~mask] = self.model_high.predict(X[~mask])
        return pred

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def run_tar_for_submercado(submercado: str):
    print(f"\n{'='*50}")
    print(f" REGIME SWITCHING (TAR) - Submercado: {submercado}")
    print(f"{'='*50}")

    df_raw = get_aligned_data(submercado)
    df_raw["pld"] = np.log(df_raw["pld"])

    df_feat = build_features(df_raw).dropna()

    target_col = "pld"
    feature_cols = [c for c in df_feat.columns if c != target_col]

    X = df_feat[feature_cols]
    y = df_feat[target_col]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_metrics = []
    
    all_dates = []
    all_y_true = []
    all_y_pred = []

    # O limiar físico: 30% de armazenamento. Abaixo disso, o risco de déficit dispara (regime de estresse).
    # Acima de 30%, o sistema opera com folga (regime normal).
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = ThresholdRegressor(threshold_col="ear_pct", threshold_val=30.0, alpha=1.0)
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test.values)
        pd.DataFrame({
            "data": X_test.index.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred
        }).to_csv(PRED_DIR / f"tar_{submercado}_fold{fold+1}.csv", sep=";", index=False)


        fold_mae = mean_absolute_error(y_true, y_pred)
        fold_rmse = root_mean_squared_error(y_true, y_pred)
        fold_mape = mape(y_true, y_pred)

        fold_metrics.append({"fold": fold + 1, "mae": fold_mae, "rmse": fold_rmse, "mape": fold_mape})
        print(f"  Fold {fold + 1} | MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f} | MAPE: {fold_mape:.2f}%")
        
        all_dates.extend(X_test.index)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_dates, all_y_true, label='Observado', color='black', alpha=0.7)
    ax.plot(all_dates, all_y_pred, label='Previsto', color='magenta')
    ax.set_title(f'Regime Switching (TAR) - Previsão de PLD ({submercado})')
    ax.set_xlabel('Data')
    ax.set_ylabel('PLD (R$/MWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'tar_forecast_{submercado}.png', dpi=150)
    plt.close(fig)

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_mape = np.mean([m["mape"] for m in fold_metrics])

    print(f"\n  MÉDIA CV (Walk-Forward {N_SPLITS}x{TEST_SIZE} dias)")
    print(f"  MAE:  {avg_mae:.2f}")
    print(f"  RMSE: {avg_rmse:.2f}")
    print(f"  MAPE: {avg_mape:.2f}%")

    return {
        "submercado": submercado,
        "mae": avg_mae, "rmse": avg_rmse, "mape": avg_mape
    }

def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_tar_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "tar_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - REGIME SWITCHING TAR (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "TEST_SIZE": TEST_SIZE,
        "N_SPLITS": N_SPLITS
    }
    
    model_params = {
        "model_type": "Threshold Autoregressive (TAR)",
        "threshold_variable": "ear_pct",
        "threshold_value": 30.0,
        "internal_regressor": "Ridge",
        "ridge_hyperparameters": {
            "alpha": 1.0
        }
    }
    
    save_reproducibility_report(
        model_name="tar_diario",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )

if __name__ == "__main__":
    main()
