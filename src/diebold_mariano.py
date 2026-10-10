import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import itertools
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "output" / "predictions"
OUTPUT_FILE = BASE_DIR / "output" / "dm_test_results.csv"

MODELS = ["naive_rw", "naive_seasonal", "sarimax", "tar", "ridge", "rf", "lightgbm", "lstm", "gru"]
SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]
FOLDS = 4

def diebold_mariano(e1, e2, h=1, criterion="MAE"):
    """
    Teste DM bilateral com correção HLN para amostras finitas.
    e1, e2: erros de previsão (actual - predicted).
    h: horizonte de previsão.
    criterion: 'MAE' (|e1| - |e2|) ou 'MSE' (e1^2 - e2^2).
    """
    if criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "MSE":
        d = e1**2 - e2**2
    else:
        raise ValueError(f"Critério '{criterion}' não suportado. Use 'MAE' ou 'MSE'.")

    T = len(d)
    
    if T == 0:
        return np.nan, np.nan
        
    d_mean = d.mean()
    gamma_0 = np.var(d, ddof=0)

    # Variância com correção de autocorrelação (Newey-West) para h > 1
    V = gamma_0
    for k in range(1, h):
        if k < T:
            gamma_k = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
            V += 2 * gamma_k
    
    # Previne divisão por zero se V for muito pequeno
    if V <= 0:
        return np.nan, np.nan
        
    V /= T
    dm_stat = d_mean / np.sqrt(V)

    # Correção HLN para amostras finitas
    hln_factor = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat_hln = dm_stat * hln_factor

    p_value = 2 * stats.t.sf(np.abs(dm_stat_hln), df=T-1)
    return dm_stat_hln, p_value

def main():
    if not PRED_DIR.exists():
        print(f"[ERRO] Diretório {PRED_DIR} não encontrado.")
        return
        
    results = []

    for sub in SUBMERCADOS:
        errors = {}
        for model in MODELS:
            e_all = []
            for fold in range(1, FOLDS + 1):
                file_path = PRED_DIR / f"{model}_{sub}_fold{fold}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path, sep=";")
                    e_all.extend(df["real"].values - df["previsto"].values)
            if e_all:
                errors[model] = np.array(e_all)
                
        # Testar todos os pares para MAE e MSE
        for loss in ["MAE", "MSE"]:
            sub_pairs = []
            p_vals = []
            for m1, m2 in itertools.combinations(MODELS, 2):
                if m1 in errors and m2 in errors:
                    stat, pval = diebold_mariano(errors[m1], errors[m2], h=1, criterion=loss)
                    sub_pairs.append({
                        "submercado": sub,
                        "modelo_1": m1,
                        "modelo_2": m2,
                        "dm_statistic": stat,
                        "p_value": pval,
                        "loss_function": loss
                    })
                    p_vals.append(pval if not np.isnan(pval) else 1.0)
            
            # Correção de Holm-Bonferroni para comparações múltiplas no submercado
            if p_vals:
                _, p_adj, _, _ = multipletests(p_vals, method="holm")
                for i, pair in enumerate(sub_pairs):
                    pair["p_value_adj"] = p_adj[i]
                    pair["significativo_5pct"] = p_adj[i] < 0.05
                    results.append(pair)

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_FILE, sep=";", index=False)
        print(f"Resultados do teste Diebold-Mariano salvos em {OUTPUT_FILE}")
    else:
        print("[AVISO] Nenhum resultado gerado para DM test.")

if __name__ == "__main__":
    main()

