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
N_SPLITS = 12

class ThresholdRegressor:
    """Modelo econométrico de Regime Switching Observável (Threshold Regression)."""
    def __init__(self, threshold_col="ear_pct", alpha=1.0, min_pct=0.15, max_pct=0.85):
        self.threshold_col = threshold_col
        self.alpha = alpha
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.model_low = Ridge(alpha=self.alpha)
        self.model_high = Ridge(alpha=self.alpha)
        self.scaler_low = StandardScaler()
        self.scaler_high = StandardScaler()
        self.best_threshold = None
        
    def fit(self, X, y):
        q = X[self.threshold_col].values
        
        # Grid search de Hansen (1999) para estimação de limiar
        percentiles = np.linspace(self.min_pct, self.max_pct, num=50)
        candidates = np.quantile(q, percentiles)
        
        best_ssr = np.inf
        self.best_threshold = candidates[0]
        
        for gamma in candidates:
            mask = q < gamma
            if mask.sum() < 5 or (~mask).sum() < 5:
                continue
                
            X_low, y_low = X[mask], y[mask]
            X_high, y_high = X[~mask], y[~mask]
            
            sc_low = StandardScaler()
            X_low_scaled = sc_low.fit_transform(X_low)
            mod_low = Ridge(alpha=self.alpha).fit(X_low_scaled, y_low)
            ssr_low = np.sum((y_low - mod_low.predict(X_low_scaled))**2)
            
            sc_high = StandardScaler()
            X_high_scaled = sc_high.fit_transform(X_high)
            mod_high = Ridge(alpha=self.alpha).fit(X_high_scaled, y_high)
            ssr_high = np.sum((y_high - mod_high.predict(X_high_scaled))**2)
            
            ssr_total = ssr_low + ssr_high
            if ssr_total < best_ssr:
                best_ssr = ssr_total
                self.best_threshold = gamma
                
        # Treinamento final com o limiar ótimo estimado
        mask_final = q < self.best_threshold
        if mask_final.sum() > 0:
            X_low_final = self.scaler_low.fit_transform(X[mask_final])
            self.model_low.fit(X_low_final, y[mask_final])
        if (~mask_final).sum() > 0:
            X_high_final = self.scaler_high.fit_transform(X[~mask_final])
            self.model_high.fit(X_high_final, y[~mask_final])
            
        return self
        
    def predict(self, X):
        pred = np.zeros(len(X))
        mask = X[self.threshold_col].values < self.best_threshold
        
        if mask.sum() > 0:
            X_low_scaled = self.scaler_low.transform(X[mask])
            pred[mask] = self.model_low.predict(X_low_scaled)
        if (~mask).sum() > 0:
            X_high_scaled = self.scaler_high.transform(X[~mask])
            pred[~mask] = self.model_high.predict(X_high_scaled)
            
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

        model = ThresholdRegressor(threshold_col="ear_pct", alpha=1.0)
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test.values)
        
        test_piso = X_test["pld_piso"].values
        test_teto = X_test["pld_teto"].values
        y_pred = np.clip(y_pred, test_piso, test_teto)
        
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
        "threshold_value": "estimated_via_hansen_grid",
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
