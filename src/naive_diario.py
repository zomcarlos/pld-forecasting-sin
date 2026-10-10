"""
Modelos Benchmark Ingênuos (Naive / Random Walk) v1.

Implementa:
1. Naive Random Walk (RW): y_hat_t = y_{t-1}
2. Naive Sazonal (S7): y_hat_t = y_{t-7}

Utilizados como baseline fundamental de acurácia preditiva.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from data_loader import SUBMERCADOS, get_aligned_data
from reproducibility import save_reproducibility_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PRED_DIR = OUTPUT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 30
N_SPLITS = 12


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run_naive_for_submercado(submercado: str):
    print(f"\n{'='*40}")
    print(f" Naive / Random Walk - Submercado: {submercado}")
    print(f"{'='*40}")

    df_raw = get_aligned_data(submercado)
    pld = df_raw["pld"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    metrics_rw = []
    metrics_s7 = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(pld)):
        test_pld = pld.iloc[test_idx]

        # Previsão RW (1 dia atrás)
        y_true = test_pld.values
        y_pred_rw = pld.iloc[np.array(test_idx) - 1].values
        
        # Previsão Sazonal (7 dias atrás)
        y_pred_s7 = pld.iloc[np.array(test_idx) - 7].values

        # Persistir previsões
        pd.DataFrame({
            "data": test_pld.index.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred_rw
        }).to_csv(PRED_DIR / f"naive_rw_{submercado}_fold{fold+1}.csv", sep=";", index=False)

        pd.DataFrame({
            "data": test_pld.index.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred_s7
        }).to_csv(PRED_DIR / f"naive_seasonal_{submercado}_fold{fold+1}.csv", sep=";", index=False)

        # Métricas RW
        mae_rw = mean_absolute_error(y_true, y_pred_rw)
        rmse_rw = root_mean_squared_error(y_true, y_pred_rw)
        mape_rw = mape(y_true, y_pred_rw)
        metrics_rw.append({"fold": fold + 1, "mae": mae_rw, "rmse": rmse_rw, "mape": mape_rw})

        # Métricas S7
        mae_s7 = mean_absolute_error(y_true, y_pred_s7)
        rmse_s7 = root_mean_squared_error(y_true, y_pred_s7)
        mape_s7 = mape(y_true, y_pred_s7)
        metrics_s7.append({"fold": fold + 1, "mae": mae_s7, "rmse": rmse_s7, "mape": mape_s7})

        print(f"  Fold {fold + 1} | RW MAE: {mae_rw:.2f} | S7 MAE: {mae_s7:.2f}")

    avg_mae_rw = float(np.mean([m["mae"] for m in metrics_rw]))
    avg_rmse_rw = float(np.mean([m["rmse"] for m in metrics_rw]))
    avg_mape_rw = float(np.mean([m["mape"] for m in metrics_rw]))

    avg_mae_s7 = float(np.mean([m["mae"] for m in metrics_s7]))
    avg_rmse_s7 = float(np.mean([m["rmse"] for m in metrics_s7]))
    avg_mape_s7 = float(np.mean([m["mape"] for m in metrics_s7]))

    return {
        "submercado": submercado,
        "rw_mae": avg_mae_rw, "rw_rmse": avg_rmse_rw, "rw_mape": avg_mape_rw,
        "s7_mae": avg_mae_s7, "s7_rmse": avg_rmse_s7, "s7_mape": avg_mape_s7,
    }


def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_naive_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "naive_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - BENCHMARKS NAIVE (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    save_reproducibility_report(
        model_name="naive_diario",
        global_params={"TEST_SIZE": TEST_SIZE, "N_SPLITS": N_SPLITS},
        model_params={"baselines": ["naive_rw (y_t-1)", "naive_seasonal (y_t-7)"]},
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade",
    )


if __name__ == "__main__":
    main()
