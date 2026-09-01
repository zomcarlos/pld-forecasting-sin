# Relatório de Reprodutibilidade e Parâmetros: gru_diario
Gerado automaticamente em: `2026-09-01T19:34:04.958018`

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
| SUL | 26.7401 | 34.2872 | 13.8388% |
| SUDESTE | 17.5287 | 23.5767 | 10.3572% |
| NORDESTE | 36.6632 | 44.3021 | 22.3690% |
| NORTE | 23.8163 | 30.6091 | 16.1756% |
