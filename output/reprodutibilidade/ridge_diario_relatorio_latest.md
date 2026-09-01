# Relatório de Reprodutibilidade e Parâmetros: ridge_diario
Gerado automaticamente em: `2026-09-01T19:31:10.970931`

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
- **model_type:** `Ridge Regression`
- **scaler:** `StandardScaler`
- **ridge_hyperparameters:**
  - *alpha:* `1.0`
  - *random_state:* `42`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | 22.1491 | 30.0382 | 11.9886% |
| SUDESTE | 18.0478 | 24.0226 | 10.8548% |
| NORDESTE | 20.9731 | 27.2727 | 14.9555% |
| NORTE | 21.5363 | 27.9218 | 15.4193% |
