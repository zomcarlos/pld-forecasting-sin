import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "output" / "predictions"
OUTPUT_DIR = BASE_DIR / "output"

MODELS = ["sarimax", "tar", "ridge", "rf", "lightgbm", "lstm", "gru"]
SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]
FOLDS = 4

def main():
    if not PRED_DIR.exists():
        print(f"[ERRO] Diretório {PRED_DIR} não encontrado.")
        return

    for model in MODELS:
        for sub in SUBMERCADOS:
            all_real, all_pred, all_dates = [], [], []
            residual_stds = []

            for fold in range(1, FOLDS + 1):
                file_path = PRED_DIR / f"{model}_{sub}_fold{fold}.csv"
                if not file_path.exists():
                    continue
                    
                df = pd.read_csv(file_path, sep=";")
                residuals = df["real"].values - df["previsto"].values
                sigma = residuals.std()
                residual_stds.append(sigma)

                all_dates.extend(pd.to_datetime(df["data"]))
                all_real.extend(df["real"])
                all_pred.extend(df["previsto"])

            if not all_pred:
                continue

            # Média dos desvios-padrão dos resíduos sobre todos os folds
            sigma_mean = np.mean(residual_stds)
            all_pred = np.array(all_pred)
            all_real = np.array(all_real)
            all_dates = np.array(all_dates)
            
            upper = all_pred + 1.96 * sigma_mean
            lower = all_pred - 1.96 * sigma_mean

            plt.figure(figsize=(12, 6))
            plt.plot(all_dates, all_real, label="Real", color="black", linewidth=1.5)
            plt.plot(all_dates, all_pred, label="Previsto", color="blue", linewidth=1.5, alpha=0.8)
            plt.fill_between(all_dates, lower, upper, color="blue", alpha=0.2, label="IC 95%")
            
            plt.title(f"Previsão com Intervalo de 95% - {model.upper()} ({sub})")
            plt.xlabel("Data")
            plt.ylabel("PLD Diário (R$/MWh)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            
            out_file = OUTPUT_DIR / f"{model}_forecast_ci_{sub}.png"
            plt.savefig(out_file, bbox_inches="tight", dpi=300)
            plt.close()
            
    print(f"Gráficos de intervalos de previsão salvos em {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
