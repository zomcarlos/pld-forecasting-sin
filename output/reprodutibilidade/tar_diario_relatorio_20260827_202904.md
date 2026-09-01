# Relatório de Reprodutibilidade e Parâmetros: tar_diario
Gerado automaticamente em: `2026-08-27T20:29:04.006040`

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
- **model_type:** `Threshold Autoregressive (TAR)`
- **threshold_variable:** `ear_pct`
- **threshold_value:** `30.0`
- **internal_regressor:** `Ridge`
- **ridge_hyperparameters:**
  - *alpha:* `1.0`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | 27.6782 | 36.4805 | 14.0157% |
| SUDESTE | 18.0771 | 24.0608 | 10.8231% |
| NORDESTE | 20.9305 | 27.3099 | 14.8854% |
| NORTE | 21.3566 | 27.8533 | 15.3856% |
