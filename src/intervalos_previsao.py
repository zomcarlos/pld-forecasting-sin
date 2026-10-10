import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "output" / "predictions"
OUTPUT_DIR = BASE_DIR / "output"

MODELS = ["naive_rw", "naive_seasonal", "sarimax", "tar", "ridge", "rf", "lightgbm", "lstm", "gru"]
SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]
FOLDS = 12

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

            all_pred = np.array(all_pred)
            all_real = np.array(all_real)
            all_dates = np.array(all_dates)
            
            # Conformal Prediction: Quantis empíricos dos resíduos fora da amostra (OOS)
            residuals_oos = all_real - all_pred
            q2_5 = np.percentile(residuals_oos, 2.5)
            q97_5 = np.percentile(residuals_oos, 97.5)
            
            upper = all_pred + q97_5
            lower = all_pred + q2_5
            
            # Limite mínimo físico para preços (sem PLD negativo estrutural)
            lower = np.clip(lower, 0.0, None)
            
            # Salvar limites de cobertura para os testes de Kupiec e Christoffersen
            df_limits = pd.DataFrame({
                "data": all_dates,
                "real": all_real,
                "previsto": all_pred,
                "limite_inferior": lower,
                "limite_superior": upper
            })
            df_limits.to_csv(PRED_DIR / f"{model}_{sub}_intervalos.csv", index=False, sep=";")

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
