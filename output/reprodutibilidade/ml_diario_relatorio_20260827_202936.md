# Relatório de Reprodutibilidade e Parâmetros: ml_diario
Gerado automaticamente em: `2026-08-27T20:29:36.960263`

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
- **model_type:** `LightGBM Regressor`
- **lgb_hyperparameters:**
  - *objective:* `regression`
  - *metric:* `mae`
  - *n_estimators:* `500`
  - *learning_rate:* `0.05`
  - *num_leaves:* `31`
  - *max_depth:* `-1`
  - *random_state:* `42`
  - *n_jobs:* `-1`
  - *early_stopping_rounds:* `30`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | 25.9853 | 32.7524 | 14.1173% |
| SUDESTE | 20.6180 | 26.2788 | 12.3509% |
| NORDESTE | 25.3031 | 31.9957 | 17.4139% |
| NORTE | 26.1920 | 32.8284 | 18.9584% |
