# Relatório de Reprodutibilidade e Parâmetros: sarimax_diario
Gerado automaticamente em: `2026-09-01T19:31:01.418922`

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
- **TEST_SIZE:** `30`
- **N_SPLITS:** `4`
- **SEASONAL_PERIOD:** `7`
- **EXOG_COLS:** `['ena', 'ear_pct', 'carga']`

## 4. Parâmetros Específicos do Modelo / Testes
- **model_type:** `SARIMAX`
- **auto_arima_selection:**
  - *max_p:* `2`
  - *max_q:* `2`
  - *max_P:* `1`
  - *max_Q:* `1`
  - *max_d:* `1`
  - *max_D:* `1`
  - *seasonal:* `True`
  - *m:* `7`
  - *stepwise:* `True`
  - *suppress_warnings:* `True`
  - *error_action:* `ignore`
- **sarimax_fit:**
  - *enforce_stationarity:* `False`
  - *enforce_invertibility:* `False`
- **forecast:**
  - *alpha:* `0.05`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | 42.1536 | 52.5661 | 22.7586% |
| SUDESTE | 35.7022 | 43.4559 | 22.1757% |
| NORDESTE | 37.3838 | 43.8914 | 32.2763% |
| NORTE | 43.5978 | 52.0737 | 37.7596% |
