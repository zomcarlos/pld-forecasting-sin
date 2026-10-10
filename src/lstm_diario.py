"""
LSTM Diário v3 — Walk-Forward Validation × 4 Submercados.

Exógenas: ENA (afluência), EAR (nível reservatório %), CARGA (demanda MWmed).
Mesmas exógenas utilizadas no modelo SARIMAX para comparação justa.

Referências:
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
  Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art
  with a look into the future. IJF, 30(4), 1030-1044.
  DOI: 10.1016/j.ijforecast.2014.08.008
- Lago, J. et al. (2018). Forecasting spot electricity prices. Applied Energy, 221,
  386-405. DOI: 10.1016/j.apenergy.2018.02.069
"""

import warnings
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_loader import SUBMERCADOS, get_aligned_data, build_features
from reproducibility import save_reproducibility_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PRED_DIR = OUTPUT_DIR / 'predictions'
PRED_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 30
N_SPLITS = 12


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Máscara para evitar divisão por zero no cálculo do erro percentual
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True indica formato (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Camada linear para converter a saída oculta na previsão do log do PLD
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # A saída out contém os hidden states para todos os time steps
        out, _ = self.lstm(x)
        # Seleciona apenas o último time step (seq_len = 1) para gerar a predição
        out = self.linear(out[:, -1, :])
        return out


def run_lstm_for_submercado(submercado: str):
    print(f"\n{'='*40}")
    print(f" LSTM - Submercado: {submercado}")
    print(f"{'='*40}")

    df_raw = get_aligned_data(submercado)
    print(f"  Observações: {len(df_raw)} | Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()}")

    # Transformação logarítmica do target
    # Aplicada antes do feature engineering para que lags e médias móveis acompanhem a escala logarítmica
    pld_original = df_raw["pld"].copy()
    df_raw["pld"] = np.log(df_raw["pld"])

    df_feat = build_features(df_raw)
    # Remove as primeiras linhas com NaN geradas pelas janelas móveis
    df_feat = df_feat.dropna()

    target_col = "pld"
    feature_cols = [c for c in df_feat.columns if c != target_col]

    X = df_feat[feature_cols]
    y = df_feat[target_col]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    fold_metrics = []
    
    all_dates = []
    all_y_true = []
    all_y_pred = []
    all_rmse = []

    epochs = 100
    patience = 15
    batch_size = 32
    hidden_size = 64
    num_layers = 2
    dropout = 0.2

    # Verifica disponibilidade de GPU para aceleração, caso contrário segue em CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Valid set para early stopping
        # Separa os últimos 10% do treino como validação para monitorar overfitting fora da amostra
        val_size = int(len(X_train) * 0.1)
        X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
        X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]

        # Padronização (StandardScaler) - fit apenas no treino evita data leakage
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        seq_len = 7
        # Empilhamento temporário para montar as janelas contínuas de 7 dias
        X_all_scaled = np.vstack((X_tr_scaled, X_val_scaled, X_test_scaled))
        
        X_seq = []
        for i in range(len(X_all_scaled)):
            if i < seq_len - 1:
                pad_size = seq_len - 1 - i
                pad = np.repeat(X_all_scaled[0:1], pad_size, axis=0)
                seq = np.vstack((pad, X_all_scaled[:i+1]))
            else:
                seq = X_all_scaled[i - seq_len + 1 : i + 1]
            X_seq.append(seq)
        X_seq = np.array(X_seq)
        
        n_tr = len(X_tr_scaled)
        n_val = len(X_val_scaled)
        
        X_tr_seq = X_seq[:n_tr]
        X_val_seq = X_seq[n_tr : n_tr + n_val]
        X_test_seq = X_seq[n_tr + n_val :]

        # Conversão para tensores 3D apropriados para LSTM/BPTT: (samples, seq_len=7, features)
        X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
        
        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        
        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_global_val_loss = np.inf
        best_model_weights = None
        best_hidden_size = hidden_size

        for hs in [32, 64]:
            for lr in [0.001, 0.005]:
                cand_model = LSTMModel(input_size=len(feature_cols), hidden_size=hs, num_layers=num_layers, dropout=dropout).to(device)
                cand_criterion = nn.MSELoss()
                cand_optimizer = optim.Adam(cand_model.parameters(), lr=lr)
                cand_patience_counter = 0
                cand_best_val_loss = np.inf
                cand_best_weights = None

                for epoch in range(epochs):
                    cand_model.train()
                    for batch_X, batch_y in train_loader:
                        cand_optimizer.zero_grad()
                        outputs = cand_model(batch_X)
                        loss = cand_criterion(outputs, batch_y)
                        loss.backward()
                        cand_optimizer.step()
                    
                    cand_model.eval()
                    with torch.no_grad():
                        val_outputs = cand_model(X_val_t)
                        val_loss = cand_criterion(val_outputs, y_val_t).item()
                    
                    if val_loss < cand_best_val_loss:
                        cand_best_val_loss = val_loss
                        cand_best_weights = copy.deepcopy(cand_model.state_dict())
                        cand_patience_counter = 0
                    else:
                        cand_patience_counter += 1

                    if cand_patience_counter >= patience:
                        break

                if cand_best_val_loss < best_global_val_loss:
                    best_global_val_loss = cand_best_val_loss
                    best_model_weights = cand_best_weights
                    best_hidden_size = hs

        model = LSTMModel(input_size=len(feature_cols), hidden_size=best_hidden_size, num_layers=num_layers, dropout=dropout).to(device)
        model.load_state_dict(best_model_weights)
        model.eval()
        with torch.no_grad():
            y_pred_log_t = model(X_test_t)
        
        y_pred_log = y_pred_log_t.cpu().numpy().flatten()
        
        # Reverte a transformação logarítmica para voltar aos valores monetários reais (R$/MWh)
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test.values)
        
        test_piso = X_test["pld_piso"].values
        test_teto = X_test["pld_teto"].values
        y_pred = np.clip(y_pred, test_piso, test_teto)
        
        pd.DataFrame({
            "data": X_test.index.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred
        }).to_csv(PRED_DIR / f"lstm_{submercado}_fold{fold+1}.csv", sep=";", index=False)


        fold_mae = mean_absolute_error(y_true, y_pred)
        fold_rmse = root_mean_squared_error(y_true, y_pred)
        fold_mape = mape(y_true, y_pred)

        fold_metrics.append({"fold": fold + 1, "mae": fold_mae, "rmse": fold_rmse, "mape": fold_mape})
        print(f"  Fold {fold + 1} | MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f} | MAPE: {fold_mape:.2f}%")
        
        all_dates.extend(X_test.index)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_rmse.extend([fold_rmse] * len(y_test))

    all_y_pred = np.array(all_y_pred)
    all_rmse = np.array(all_rmse)
    
    # Gráfico de previsão vs valor real para o submercado em análise
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_dates, all_y_true, label='Observado', color='black', alpha=0.7)
    ax.plot(all_dates, all_y_pred, label='Previsto', color='purple')
    # Faixa de incerteza (±RMSE) atua como uma aproximação visual da margem de erro
    ax.fill_between(
        all_dates,
        np.maximum(0, all_y_pred - all_rmse),
        all_y_pred + all_rmse,
        color='purple', alpha=0.2, label='Margem de Erro (± RMSE)'
    )
    ax.set_title(f'LSTM - Previsão de PLD ({submercado})')
    ax.set_xlabel('Data')
    ax.set_ylabel('PLD (R$/MWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'lstm_forecast_{submercado}.png', dpi=150)
    plt.close(fig)

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_mape = np.mean([m["mape"] for m in fold_metrics])

    print(f"\n  MÉDIA CV (Walk-Forward {N_SPLITS}x{TEST_SIZE} dias)")
    print(f"  MAE:  {avg_mae:.2f}")
    print(f"  RMSE: {avg_rmse:.2f}")
    print(f"  MAPE: {avg_mape:.2f}%")

    return {"submercado": submercado, "mae": avg_mae, "rmse": avg_rmse, "mape": avg_mape}


def main():
    results = []
    for sub in SUBMERCADOS:
        res = run_lstm_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "lstm_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - LSTM (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "TEST_SIZE": TEST_SIZE,
        "N_SPLITS": N_SPLITS
    }
    
    model_params = {
        "model_type": "LSTM Neural Network",
        "lstm_hyperparameters": {
            "epochs": 100,
            "patience": 15,
            "batch_size": 32,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_function": "MSELoss"
        }
    }
    
    save_reproducibility_report(
        model_name="lstm_diario",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )


if __name__ == "__main__":
    main()
