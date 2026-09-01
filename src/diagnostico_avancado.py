import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "output" / "predictions"
OUTPUT_DIR = BASE_DIR / "output" / "diagnosticos"
OUTPUT_FILE = OUTPUT_DIR / "residuals_advanced.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["sarimax", "tar", "ridge", "rf", "lightgbm", "lstm", "gru"]
SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]
FOLDS = 4

def main():
    if not PRED_DIR.exists():
        print(f"[ERRO] Diretório {PRED_DIR} não encontrado.")
        return

    results = []

    for model in MODELS:
        for sub in SUBMERCADOS:
            e_all = []
            for fold in range(1, FOLDS + 1):
                file_path = PRED_DIR / f"{model}_{sub}_fold{fold}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path, sep=";")
                    e_all.extend(df["real"].values - df["previsto"].values)
            
            if not e_all:
                continue
                
            resid = np.array(e_all)
            
            # 1. Jarque-Bera (Normalidade)
            jb_stat, jb_pval = stats.jarque_bera(resid)
            results.append({
                "modelo": model,
                "submercado": sub,
                "teste": "Jarque-Bera",
                "estatistica": jb_stat,
                "p_valor": jb_pval,
                "rejeita_5pct": jb_pval < 0.05
            })
            
            # 2. ARCH-LM (Heterocedasticidade)
            arch_stat, arch_pval, _, _ = het_arch(resid, nlags=7)
            results.append({
                "modelo": model,
                "submercado": sub,
                "teste": "ARCH-LM (lag=7)",
                "estatistica": arch_stat,
                "p_valor": arch_pval,
                "rejeita_5pct": arch_pval < 0.05
            })
            
            # 3. Ljung-Box (Autocorrelação)
            lb_res = acorr_ljungbox(resid, lags=[7], return_df=True)
            lb_stat = lb_res.iloc[0]["lb_stat"]
            lb_pval = lb_res.iloc[0]["lb_pvalue"]
            results.append({
                "modelo": model,
                "submercado": sub,
                "teste": "Ljung-Box (lag=7)",
                "estatistica": lb_stat,
                "p_valor": lb_pval,
                "rejeita_5pct": lb_pval < 0.05
            })
            
            # QQ-plot
            plt.figure(figsize=(6, 6))
            stats.probplot(resid, dist="norm", plot=plt)
            plt.title(f"QQ-Plot Resíduos - {model.upper()} ({sub})")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"qqplot_{model}_{sub}.png", dpi=150)
            plt.close()

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_FILE, sep=";", index=False)
        print(f"Diagnóstico avançado de resíduos salvo em {OUTPUT_FILE}")
        print(f"QQ-plots salvos em {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
