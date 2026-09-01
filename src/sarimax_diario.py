"""
SARIMAX Diário v3 — Walk-Forward Validation × 4 Submercados.

Exógenas: ENA (afluência), EAR (nível reservatório %), CARGA (demanda MWmed).
Mesmas exógenas utilizadas no modelo ML para comparação justa.

Referências:
- Box, G. et al. (2015). Time Series Analysis. 5ª ed. Wiley. ISBN: 978-1-118-67502-1
- Hyndman, R.; Khandakar, Y. (2008). Automatic time series forecasting: the forecast
  package for R. JSS, 27(3). DOI: 10.18637/jss.v027.i03
- Pereira, T.P. (2021). SARIMA/SARIMAX para demanda de energia. FURG.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from data_loader import SUBMERCADOS, get_aligned_data
from reproducibility import save_reproducibility_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Horizonte de 30 dias por fold simula o planejamento operacional mensal
TEST_SIZE = 30
N_SPLITS = 4
# Sazonalidade semanal (padrão de consumo dias úteis vs. fins de semana)
SEASONAL_PERIOD = 7

# Mesmas variáveis exógenas dos modelos ML para viabilizar comparação justa
EXOG_COLS = ["ena", "ear_pct", "carga"]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Máscara evita divisão por zero caso o PLD atinja o piso regulatório
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run_sarimax_for_submercado(submercado: str):
    print(f"\n{'='*50}")
    print(f" SARIMAX - Submercado: {submercado}")
    print(f"{'='*50}")

    df_raw = get_aligned_data(submercado)
    print(f"  Observações: {len(df_raw)} | Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()}")

    # Transformação logarítmica do target
    # Estabiliza a variância da série temporal frente à alta volatilidade do PLD
    pld_log = np.log(df_raw["pld"])
    exog = df_raw[EXOG_COLS]

    # Validação temporal sem embaralhamento para evitar vazamento de dados (data leakage)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_metrics = []
    selected_order = None
    selected_seasonal = None
    
    all_dates = []
    all_y_true = []
    all_y_pred = []
    all_y_lower = []
    all_y_upper = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(pld_log)):
        train_pld = pld_log.iloc[train_idx]
        test_pld = pld_log.iloc[test_idx]
        train_exog = exog.iloc[train_idx]
        test_exog = exog.iloc[test_idx]

        if selected_order is None:
            # Seleção de ordem no primeiro fold
            # Identificação única economiza custo computacional assumindo estabilidade da estrutura temporal
            print("  Seleção de ordem (auto_arima)...")
            auto_model = auto_arima(
                train_pld,
                exogenous=train_exog.values,
                seasonal=True,
                m=SEASONAL_PERIOD,
                stepwise=True,
                max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
                trace=False,
                suppress_warnings=True,
                error_action="ignore",
            )
            selected_order = auto_model.order
            selected_seasonal = auto_model.seasonal_order
            print(f"  Ordem selecionada: SARIMAX{selected_order}x{selected_seasonal}")

        # Ajustar modelo com a ordem fixa em cada fold
        model = SARIMAX(
            train_pld,
            exog=train_exog.values,
            order=selected_order,
            seasonal_order=selected_seasonal,
            # Desativa restrições estritas para evitar falhas de convergência próximas ao círculo unitário
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

        # Previsão pontual e intervalos de confiança de 95% (alpha=0.05)
        forecast_result = fitted.get_forecast(
            steps=TEST_SIZE,
            exog=test_exog.values,
        )
        y_pred_log = forecast_result.predicted_mean.values
        ci_log = forecast_result.conf_int(alpha=0.05)

        # Reverter log
        # Retorno à escala original (R$/MWh) para cálculo das métricas de erro
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(test_pld.values)
        y_lower = np.exp(ci_log.iloc[:, 0].values)
        y_upper = np.exp(ci_log.iloc[:, 1].values)

        fold_mae = mean_absolute_error(y_true, y_pred)
        fold_rmse = root_mean_squared_error(y_true, y_pred)
        fold_mape = mape(y_true, y_pred)

        fold_metrics.append({"fold": fold + 1, "mae": fold_mae, "rmse": fold_rmse, "mape": fold_mape})
        print(f"  Fold {fold + 1} | MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f} | MAPE: {fold_mape:.2f}%")
        
        all_dates.extend(test_pld.index)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_lower.extend(y_lower)
        all_y_upper.extend(y_upper)

    # Gráfico de previsão vs real
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_dates, all_y_true, label='Observado', color='black', alpha=0.7)
    ax.plot(all_dates, all_y_pred, label='Previsto', color='blue')
    ax.fill_between(all_dates, all_y_lower, all_y_upper, color='blue', alpha=0.2, label='Margem de Erro (95%)')
    ax.set_title(f'SARIMAX - Previsão de PLD ({submercado})')
    ax.set_xlabel('Data')
    ax.set_ylabel('PLD (R$/MWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'sarimax_forecast_{submercado}.png', dpi=150)
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
        "mae": avg_mae, "rmse": avg_rmse, "mape": avg_mape,
        "order": f"{selected_order}{selected_seasonal}",
    }


def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_sarimax_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "sarimax_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - SARIMAX (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "TEST_SIZE": TEST_SIZE,
        "N_SPLITS": N_SPLITS,
        "SEASONAL_PERIOD": SEASONAL_PERIOD,
        "EXOG_COLS": EXOG_COLS
    }
    
    model_params = {
        "model_type": "SARIMAX",
        "auto_arima_selection": {
            "max_p": 2, "max_q": 2, "max_P": 1, "max_Q": 1, "max_d": 1, "max_D": 1,
            "seasonal": True, "m": SEASONAL_PERIOD, "stepwise": True,
            "suppress_warnings": True, "error_action": "ignore"
        },
        "sarimax_fit": {
            "enforce_stationarity": False,
            "enforce_invertibility": False
        },
        "forecast": {
            "alpha": 0.05
        }
    }
    
    save_reproducibility_report(
        model_name="sarimax_diario",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )


if __name__ == "__main__":
    main()

## GRETL -> software para verificação
## ALUNO OUVINTE MESTRADO SÉRIES TEMPORAIS