# Relatório de Reprodutibilidade e Parâmetros: diagnostico_estatistico
Gerado automaticamente em: `2026-08-27T20:25:21.356771`

## 1. Ambiente do Sistema
- **Sistema Operacional:** Linux (Release: 7.0.0-30-generic)
- **Python:** `3.14.4 (main, Jun 18 2026, 14:25:02) [GCC 15.2.0]`

## 2. Dependências e Versões das Bibliotecas
- **numpy:** `2.5.2`
- **pandas:** `3.0.5`
- **sklearn:** `1.9.0`
- **statsmodels:** `0.14.6`
- **pmdarima:** `2.1.1`
- **lightgbm:** `4.7.0`
- **torch:** `2.13.0+cpu`

## 3. Parâmetros Globais de Execução
- **SEASONAL_PERIOD:** `7`
- **EXOG_COLS:** `['ena', 'ear_pct', 'carga']`

## 4. Parâmetros Específicos do Modelo / Testes
- **adf_test:**
  - *autolag:* `AIC`
  - *regression:* `c`
- **kpss_test:**
  - *regression:* `ct`
  - *nlags:* `auto`
- **zivot_andrews_test:**
  - *autolag:* `AIC`
- **auto_arima:**
  - *seasonal:* `True`
  - *m:* `7`
  - *stepwise:* `True`
  - *max_p:* `2`
  - *max_q:* `2`
  - *max_P:* `1`
  - *max_Q:* `1`
  - *max_d:* `1`
  - *max_D:* `1`
  - *test:* `kpss`
  - *suppress_warnings:* `True`
  - *error_action:* `ignore`
- **sarimax:**
  - *enforce_stationarity:* `False`
  - *enforce_invertibility:* `False`
- **ljung_box:**
  - *lags:* `[7, 14, 21, 30]`
- **acf_pacf:**
  - *lags:* `40`
- **train_split:** `0.8`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | N/A | N/A | N/A |
| SUDESTE | N/A | N/A | N/A |
| NORDESTE | N/A | N/A | N/A |
| NORTE | N/A | N/A | N/A |
