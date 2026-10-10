import pandas as pd
import numpy as np
from scipy.stats import chi2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "output" / "predictions"
OUTPUT_DIR = BASE_DIR / "output"

MODELS = ["naive_rw", "naive_seasonal", "sarimax", "tar", "ridge", "rf", "lightgbm", "lstm", "gru"]
SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]

def kupiec_pof_test(I, alpha=0.05):
    """
    Kupiec's Proportion of Failures (POF) test para cobertura incondicional.
    I: array binário de violações (1 se violou, 0 se contido)
    """
    N = len(I)
    x = I.sum()
    p_hat = x / N

    if x == 0:
        lr_pof = -2 * N * np.log(1 - alpha)
    elif x == N:
        lr_pof = -2 * N * np.log(alpha)
    else:
        num = (alpha**x) * ((1 - alpha)**(N - x))
        den = (p_hat**x) * ((1 - p_hat)**(N - x))
        # Para evitar problemas numéricos extremos
        num = np.clip(num, 1e-100, None)
        den = np.clip(den, 1e-100, None)
        lr_pof = -2 * np.log(num / den)

    # p-value da distribuição Qui-Quadrado com 1 grau de liberdade
    p_value = 1 - chi2.cdf(lr_pof, df=1)
    return p_value

def christoffersen_ind_test(I):
    """
    Christoffersen's Independence test para cobertura condicional.
    I: array binário de violações
    """
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(I)):
        if I[t-1] == 0 and I[t] == 0:
            n00 += 1
        elif I[t-1] == 0 and I[t] == 1:
            n01 += 1
        elif I[t-1] == 1 and I[t] == 0:
            n10 += 1
        elif I[t-1] == 1 and I[t] == 1:
            n11 += 1

    total_transitions = n00 + n01 + n10 + n11
    if total_transitions == 0:
        return np.nan

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / total_transitions

    num = (pi**(n01 + n11)) * ((1 - pi)**(n00 + n10))
    den = (pi01**n01) * ((1 - pi01)**n00) * (pi11**n11) * ((1 - pi11)**n10)
    
    # Prevenção numéricas contra log(0)
    num = max(num, 1e-100)
    den = max(den, 1e-100)
    
    lr_ind = -2 * np.log(num / den)
    p_value = 1 - chi2.cdf(lr_ind, df=1)
    return p_value

def main():
    if not PRED_DIR.exists():
        print(f"[ERRO] Diretório {PRED_DIR} não encontrado.")
        return

    results = []

    for model in MODELS:
        for sub in SUBMERCADOS:
            file_path = PRED_DIR / f"{model}_{sub}_intervalos.csv"
            if not file_path.exists():
                continue

            df = pd.read_csv(file_path, sep=";")
            
            # Identificação formal de Violações (I_t = 1)
            # Ocorrem quando o valor real fura o teto OU o chão do intervalo de confiança
            is_violation = (df["real"] > df["limite_superior"]) | (df["real"] < df["limite_inferior"])
            I = is_violation.astype(int).values

            N = len(I)
            x = I.sum()
            cobertura_real = (1 - (x / N)) * 100

            pval_kupiec = kupiec_pof_test(I, alpha=0.05)
            pval_christoffersen = christoffersen_ind_test(I)

            # Para alpha_teste = 0.05, passamos (aceitamos H0) se p-value > 0.05
            passou_kupiec = pval_kupiec > 0.05
            passou_christ = pval_christoffersen > 0.05 if not np.isnan(pval_christoffersen) else True

            results.append({
                "modelo": model,
                "submercado": sub,
                "observacoes": N,
                "violacoes": x,
                "cobertura_real_%": round(cobertura_real, 2),
                "kupiec_p_value": round(pval_kupiec, 4),
                "passou_kupiec": passou_kupiec,
                "christoffersen_p_value": round(pval_christoffersen, 4) if not np.isnan(pval_christoffersen) else None,
                "passou_christoffersen": passou_christ
            })

    df_results = pd.DataFrame(results)
    out_file = OUTPUT_DIR / "backtest_cobertura_intervalos.csv"
    df_results.to_csv(out_file, index=False, sep=";")
    print(f"Relatório de backtesting salvo em: {out_file}")

if __name__ == "__main__":
    main()
