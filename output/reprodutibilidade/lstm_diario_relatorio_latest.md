# Relatório de Reprodutibilidade e Parâmetros: lstm_diario
Gerado automaticamente em: `2026-09-01T19:33:02.561749`

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
- **model_type:** `LSTM Neural Network`
- **lstm_hyperparameters:**
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
| SUL | 28.4083 | 35.1719 | 14.9701% |
| SUDESTE | 21.0456 | 26.8025 | 11.9585% |
| NORDESTE | 48.8174 | 56.2091 | 30.3917% |
| NORTE | 23.4575 | 30.6413 | 15.1078% |
