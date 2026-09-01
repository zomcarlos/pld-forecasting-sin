# Relatório de Reprodutibilidade e Parâmetros: lstm_diario
Gerado automaticamente em: `2026-08-27T20:31:03.924329`

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
| SUL | 26.9850 | 34.4699 | 14.2457% |
| SUDESTE | 23.1482 | 29.0064 | 13.0416% |
| NORDESTE | 42.7691 | 51.7588 | 27.1321% |
| NORTE | 22.4811 | 29.7438 | 14.9309% |
