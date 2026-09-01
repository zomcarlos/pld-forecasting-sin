# Relatório de Reprodutibilidade e Parâmetros: gru_diario
Gerado automaticamente em: `2026-08-27T20:32:09.268000`

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
- **model_type:** `GRU Neural Network`
- **gru_hyperparameters:**
  - *epochs:* `100`
  - *patience:* `15`
  - *batch_size:* `32`
  - *hidden_size:* `64`
  - *num_layers:* `2`
  - *dropout:* `0.2`
  - *learning_rate:* `0.001`
  - *optimizer:* `Adam`
  - *loss_function:* `MSELoss`

## 5. Resultados e Métricas Obtidas
| Submercado | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| SUL | 25.4323 | 32.6538 | 13.3363% |
| SUDESTE | 19.1663 | 24.7697 | 11.2462% |
| NORDESTE | 33.2529 | 41.7655 | 20.6968% |
| NORTE | 24.1354 | 30.7471 | 16.4522% |
