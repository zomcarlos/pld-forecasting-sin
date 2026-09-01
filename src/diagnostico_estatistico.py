"""
Diagnóstico Estatístico da Série Temporal do PLD v3.

Gera relatório completo de testes estatísticos exigidos para validação
da modelagem SARIMAX, incluindo:
  1. Teste de Estacionariedade (KPSS + ADF)
  2. Teste de Raiz Unitária / Quebra Estrutural (Zivot-Andrews)
  3. Teste de Ruído Branco nos Resíduos (Ljung-Box)
  4. Análise de Autocorrelação e Médias Móveis (ACF / PACF)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from data_loader import SUBMERCADOS, get_aligned_data
from reproducibility import save_reproducibility_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "diagnosticos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIOD = 7
EXOG_COLS = ["ena", "ear_pct", "carga"]


def run_diagnostics(submercado: str):
    print(f"\n{'='*60}")
    print(f"  DIAGNÓSTICO ESTATÍSTICO — Submercado: {submercado}")
    print(f"{'='*60}")

    df_raw = get_aligned_data(submercado)
    pld_log = np.log(df_raw["pld"])
    exog = df_raw[EXOG_COLS]

    # ---------------------------------------------------------------
    # 1. TESTES DE ESTACIONARIEDADE (série em nível e diferenciada)
    # ---------------------------------------------------------------
    print("\n  [1] TESTES DE ESTACIONARIEDADE")
    print("  " + "-"*50)

    # Série original (log)
    adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(pld_log, autolag="AIC")
    kpss_stat, kpss_pval, _, kpss_crit = kpss(pld_log, regression="ct", nlags="auto")

    print(f"  Série log(PLD) em nível:")
    print(f"    ADF  — Estatística: {adf_stat:.4f} | p-valor: {adf_pval:.4f}")
    print(f"           H0: série possui raiz unitária (não-estacionária)")
    print(f"           Resultado: {'Rejeita H0 → Estacionária' if adf_pval < 0.05 else 'NÃO rejeita H0 → Não-estacionária'}")
    print(f"    KPSS — Estatística: {kpss_stat:.4f} | p-valor: {kpss_pval:.4f}")
    print(f"           H0: série é estacionária")
    print(f"           Resultado: {'Rejeita H0 → Não-estacionária' if kpss_pval < 0.05 else 'NÃO rejeita H0 → Estacionária'}")

    # Série diferenciada (d=1)
    pld_diff = pld_log.diff().dropna()
    adf_d_stat, adf_d_pval, _, _, _, _ = adfuller(pld_diff, autolag="AIC")
    kpss_d_stat, kpss_d_pval, _, _ = kpss(pld_diff, regression="ct", nlags="auto")

    print(f"\n  Série Δlog(PLD) (1ª diferença):")
    print(f"    ADF  — Estatística: {adf_d_stat:.4f} | p-valor: {adf_d_pval:.6f}")
    print(f"           Resultado: {'Rejeita H0 → Estacionária' if adf_d_pval < 0.05 else 'NÃO rejeita H0 → Não-estacionária'}")
    print(f"    KPSS — Estatística: {kpss_d_stat:.4f} | p-valor: {kpss_d_pval:.4f}")
    print(f"           Resultado: {'Rejeita H0 → Não-estacionária' if kpss_d_pval < 0.05 else 'NÃO rejeita H0 → Estacionária'}")

    # ---------------------------------------------------------------
    # 2. TESTE DE QUEBRA ESTRUTURAL / RAIZ UNITÁRIA (Zivot-Andrews)
    # ---------------------------------------------------------------
    print("\n  [2] TESTE DE QUEBRA ESTRUTURAL (Zivot-Andrews)")
    print("  " + "-"*50)

    try:
        from statsmodels.tsa.stattools import zivot_andrews
        za_result = zivot_andrews(pld_log.values, autolag="AIC")
        za_stat = za_result[0]
        za_pval = za_result[1]
        za_breakpoint = za_result[4]
        break_date = pld_log.index[za_breakpoint]
        print(f"    Estatística: {za_stat:.4f} | p-valor: {za_pval:.4f}")
        print(f"    H0: a série possui raiz unitária (sem quebra estrutural)")
        print(f"    Resultado: {'Rejeita H0 → Estacionária com quebra' if za_pval < 0.05 else 'NÃO rejeita H0 → Raiz unitária presente'}")
        print(f"    Ponto de quebra estimado: {break_date.date()}")
    except ImportError:
        print("    Teste Zivot-Andrews não disponível nesta versão do statsmodels.")
    except Exception as e:
        print(f"    Erro ao executar Zivot-Andrews: {e}")

    # ---------------------------------------------------------------
    # 3. AJUSTAR SARIMAX E TESTAR RESÍDUOS
    # ---------------------------------------------------------------
    print("\n  [3] AJUSTE DO MODELO E DIAGNÓSTICO DE RESÍDUOS")
    print("  " + "-"*50)

    # Usar 80% dos dados para ajuste do modelo diagnóstico
    n_train = int(len(pld_log) * 0.8)
    train_pld = pld_log.iloc[:n_train]
    train_exog = exog.iloc[:n_train]

    # Seleção automática de ordem
    print("  Executando auto_arima...")
    auto_model = auto_arima(
        train_pld,
        exogenous=train_exog.values,
        seasonal=True,
        m=SEASONAL_PERIOD,
        stepwise=True,
        max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
        test="kpss",  # Teste de estacionariedade utilizado pelo auto_arima
        trace=False,
        suppress_warnings=True,
        error_action="ignore",
    )
    order = auto_model.order
    seasonal = auto_model.seasonal_order
    print(f"  Ordem selecionada: SARIMAX{order}x{seasonal}")
    print(f"  AIC: {auto_model.aic():.2f}")
    print(f"  Teste de estacionariedade interno: KPSS (padrão do pmdarima)")

    # Ajustar modelo final
    model = SARIMAX(
        train_pld,
        exog=train_exog.values,
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    residuals = fitted.resid

    # ---------------------------------------------------------------
    # 3a. TESTE DE RUÍDO BRANCO (Ljung-Box nos resíduos)
    # ---------------------------------------------------------------
    print("\n  [3a] TESTE DE RUÍDO BRANCO (Ljung-Box)")
    print("  " + "-"*50)

    # Testa autocorrelação nos resíduos em diferentes lags
    lb_results = acorr_ljungbox(residuals, lags=[7, 14, 21, 30], return_df=True)
    print(f"    H0: os resíduos são ruído branco (independentes)")
    print(f"    Se p-valor > 0.05 → resíduos são ruído branco (modelo bem especificado)\n")
    print(f"    {'Lag':>6} | {'Estatística Q':>15} | {'p-valor':>10} | {'Resultado':>25}")
    print(f"    {'-'*6}-+-{'-'*15}-+-{'-'*10}-+-{'-'*25}")
    for lag_val, row in lb_results.iterrows():
        p = row["lb_pvalue"]
        status = "Ruído Branco ✓" if p > 0.05 else "Autocorrelação Residual ✗"
        print(f"    {lag_val:>6} | {row['lb_stat']:>15.4f} | {p:>10.4f} | {status:>25}")

    # ---------------------------------------------------------------
    # 3b. VERIFICAÇÃO DE MÉDIAS MÓVEIS (componentes MA do modelo)
    # ---------------------------------------------------------------
    print(f"\n  [3b] COMPONENTES DE MÉDIAS MÓVEIS")
    print("  " + "-"*50)
    print(f"    Ordem MA não-sazonal (q): {order[2]}")
    print(f"    Ordem MA sazonal (Q):     {seasonal[2]}")

    if order[2] > 0:
        ma_params = {k: v for k, v in fitted.params.items() if "ma." in k.lower() and "sigma" not in k.lower()}
        print(f"    Coeficientes MA estimados:")
        for name, val in ma_params.items():
            print(f"      {name}: {val:.6f}")
    else:
        print("    Nenhum componente MA não-sazonal no modelo selecionado.")

    # ---------------------------------------------------------------
    # 4. GRÁFICOS DE DIAGNÓSTICO
    # ---------------------------------------------------------------
    print(f"\n  [4] GERANDO GRÁFICOS DE DIAGNÓSTICO...")

    # 4a. ACF e PACF da série diferenciada
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(pld_diff, lags=40, ax=axes[0], title=f"ACF — Δlog(PLD) ({submercado})")
    plot_pacf(pld_diff, lags=40, ax=axes[1], title=f"PACF — Δlog(PLD) ({submercado})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"acf_pacf_{submercado}.png", dpi=150)
    plt.close(fig)

    # 4b. Diagnóstico dos resíduos (4 painéis)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Resíduos ao longo do tempo
    axes[0, 0].plot(residuals.index, residuals.values, alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Resíduos ao Longo do Tempo")
    axes[0, 0].set_ylabel("Resíduo")

    # Histograma dos resíduos
    axes[0, 1].hist(residuals.values, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0, 1].set_title("Distribuição dos Resíduos")
    axes[0, 1].set_xlabel("Resíduo")
    axes[0, 1].set_ylabel("Densidade")

    # ACF dos resíduos
    plot_acf(residuals, lags=30, ax=axes[1, 0], title="ACF dos Resíduos")

    # QQ-Plot dos resíduos
    from scipy import stats
    stats.probplot(residuals.values, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ-Plot dos Resíduos")

    fig.suptitle(f"Diagnóstico de Resíduos — SARIMAX ({submercado})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"diagnostico_residuos_{submercado}.png", dpi=150)
    plt.close(fig)

    # 4c. Série original com ponto de quebra (se detectado)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pld_log.index, pld_log.values, alpha=0.7, linewidth=0.5, color="black")
    ax.set_title(f"Série log(PLD) — {submercado}")
    ax.set_ylabel("log(PLD)")
    ax.set_xlabel("Data")
    ax.grid(True, linestyle="--", alpha=0.3)
    try:
        if za_pval < 0.05:
            ax.axvline(x=break_date, color="red", linestyle="--", linewidth=2, label=f"Quebra: {break_date.date()}")
            ax.legend()
    except NameError:
        pass
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"serie_quebra_{submercado}.png", dpi=150)
    plt.close(fig)

    print(f"  Gráficos salvos em: {OUTPUT_DIR}/")

    return {
        "submercado": submercado,
        "adf_pval_nivel": adf_pval,
        "kpss_pval_nivel": kpss_pval,
        "adf_pval_diff": adf_d_pval,
        "kpss_pval_diff": kpss_d_pval,
        "ordem": f"{order}x{seasonal}",
        "aic": auto_model.aic(),
        "ljungbox_lag7_pval": lb_results.loc[7, "lb_pvalue"],
        "ljungbox_lag14_pval": lb_results.loc[14, "lb_pvalue"],
    }


def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_diagnostics(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "diagnosticos_estatisticos.csv", index=False, sep=";")

    print(f"\n{'='*60}")
    print("RESUMO FINAL — DIAGNÓSTICOS ESTATÍSTICOS")
    print(f"{'='*60}")
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "SEASONAL_PERIOD": SEASONAL_PERIOD,
        "EXOG_COLS": EXOG_COLS
    }
    
    model_params = {
        "adf_test": {
            "autolag": "AIC",
            "regression": "c"
        },
        "kpss_test": {
            "regression": "ct",
            "nlags": "auto"
        },
        "zivot_andrews_test": {
            "autolag": "AIC"
        },
        "auto_arima": {
            "seasonal": True,
            "m": SEASONAL_PERIOD,
            "stepwise": True,
            "max_p": 2,
            "max_q": 2,
            "max_P": 1,
            "max_Q": 1,
            "max_d": 1,
            "max_D": 1,
            "test": "kpss",
            "suppress_warnings": True,
            "error_action": "ignore"
        },
        "sarimax": {
            "enforce_stationarity": False,
            "enforce_invertibility": False
        },
        "ljung_box": {
            "lags": [7, 14, 21, 30]
        },
        "acf_pacf": {
            "lags": 40
        },
        "train_split": 0.8
    }
    
    save_reproducibility_report(
        model_name="diagnostico_estatistico",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )


if __name__ == "__main__":
    main()
