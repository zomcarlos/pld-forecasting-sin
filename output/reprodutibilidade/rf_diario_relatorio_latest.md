# Relatório de Reprodutibilidade e Parâmetros: rf_diario
Gerado automaticamente em: `2026-09-01T19:31:30.912480`

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

## 4. Parâmetros Específicos do Modelo / Testes
- **model_type:** `Random Forest Regressor`
- **rf_hyperparameters:**
  - *n_estimators:* `200`
  - *random_state:* `42`
  - *n_jobs:* `-1`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | 23.5747 | 30.7954 | 13.6045% |
| SUDESTE | 19.4732 | 24.7648 | 12.3394% |
| NORDESTE | 23.2291 | 29.5458 | 16.6894% |
| NORTE | 24.0699 | 30.5560 | 18.1278% |
