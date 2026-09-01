import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "output" / "predictions"
OUTPUT_FILE = BASE_DIR / "output" / "cv_metrics.csv"

MODELS = ["sarimax", "tar", "ridge", "rf", "lightgbm", "lstm", "gru"]
SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]
FOLDS = 4

def compute_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    results = []
    
    if not PRED_DIR.exists():
        print(f"[ERRO] Diretório {PRED_DIR} não encontrado. Execute os modelos primeiro.")
        return

    for model in MODELS:
        for sub in SUBMERCADOS:
            sub_metrics = []
            for fold in range(1, FOLDS + 1):
                file_path = PRED_DIR / f"{model}_{sub}_fold{fold}.csv"
                if not file_path.exists():
                    print(f"[AVISO] Faltando {file_path.name}")
                    continue
                
                df = pd.read_csv(file_path, sep=";")
                mae = mean_absolute_error(df["real"], df["previsto"])
                rmse = root_mean_squared_error(df["real"], df["previsto"])
                mape = compute_mape(df["real"], df["previsto"])
                
                row = {
                    "modelo": model,
                    "submercado": sub,
                    "fold": str(fold),
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape
                }
                results.append(row)
                sub_metrics.append((mae, rmse, mape))
            
            if sub_metrics:
                mean_mae = np.mean([m[0] for m in sub_metrics])
                mean_rmse = np.mean([m[1] for m in sub_metrics])
                mean_mape = np.mean([m[2] for m in sub_metrics])
                
                results.append({
                    "modelo": model,
                    "submercado": sub,
                    "fold": "media",
                    "mae": mean_mae,
                    "rmse": mean_rmse,
                    "mape": mean_mape
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, sep=";", index=False)
    print(f"Métricas por fold salvas em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
