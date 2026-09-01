"""
GRU Diário v3 — Walk-Forward Validation × 4 Submercados.

Exógenas: ENA (afluência), EAR (nível reservatório %), CARGA (demanda MWmed).
Mesmas exógenas utilizadas no modelo SARIMAX para comparação justa.

Referências:
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).
  Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.
  EMNLP 2014. https://doi.org/10.3115/v1/D14-1179
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
N_SPLITS = 4


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Máscara para evitar divisão por zero no cálculo do erro percentual
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True indica formato (batch, seq, feature)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Camada linear para converter a saída oculta na previsão do log do PLD
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # A saída out contém os hidden states para todos os time steps
        out, _ = self.gru(x)
        # Seleciona apenas o último time step (seq_len = 1) para gerar a predição
        out = self.linear(out[:, -1, :])
        return out


def run_gru_for_submercado(submercado: str):
    print(f"\n{'='*40}")
    print(f" GRU - Submercado: {submercado}")
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

        # Reshape para tensor 3D: (samples, seq_len=1, features)
        # Necessário pois a GRU aguarda um dado sequencial, aqui interpretamos o vetor tubular instantâneo como seq_len=1
        X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32).unsqueeze(1).to(device)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device)
        
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)
        
        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        model = GRUModel(input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = np.inf
        patience_counter = 0
        best_model_weights = None
        
        train_losses = []
        val_losses = []

        # Loop de treinamento por épocas
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)
            
            epoch_train_loss /= len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            # Avaliação no conjunto de validação
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_losses.append(val_loss)
            
            # Early Stopping interrompe se a perda na validação não melhorar após "patience" rodadas
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Gera gráfico mostrando a evolução da perda (treinamento e validação) apenas no último fold
        if fold == N_SPLITS - 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(train_losses, label='Treinamento (Train)')
            ax.plot(val_losses, label='Validação (Val)')
            ax.set_title(f'Curvas de Perda (Loss) - GRU Fold {fold+1} ({submercado})')
            ax.set_xlabel('Épocas')
            ax.set_ylabel('MSE Loss')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f'gru_loss_{submercado}.png', dpi=150)
            plt.close(fig)
            
        # Carrega os pesos do melhor momento no early stopping
        model.load_state_dict(best_model_weights)
        model.eval()
        with torch.no_grad():
            y_pred_log_t = model(X_test_t)
        
        y_pred_log = y_pred_log_t.cpu().numpy().flatten()
        
        # Reverte a transformação logarítmica para voltar aos valores monetários reais (R$/MWh)
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test.values)
        pd.DataFrame({
            "data": X_test.index.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred
        }).to_csv(PRED_DIR / f"gru_{submercado}_fold{fold+1}.csv", sep=";", index=False)


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
    ax.plot(all_dates, all_y_pred, label='Previsto', color='orange')
    # Faixa de incerteza (±RMSE) atua como uma aproximação visual da margem de erro
    ax.fill_between(
        all_dates,
        np.maximum(0, all_y_pred - all_rmse),
        all_y_pred + all_rmse,
        color='orange', alpha=0.2, label='Margem de Erro (± RMSE)'
    )
    ax.set_title(f'GRU - Previsão de PLD ({submercado})')
    ax.set_xlabel('Data')
    ax.set_ylabel('PLD (R$/MWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'gru_forecast_{submercado}.png', dpi=150)
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
        res = run_gru_for_submercado(sub)
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "gru_comparativo_regioes.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESUMO FINAL - GRU (Walk-Forward Validation)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # --- REGISTRO DE REPRODUTIBILIDADE ---
    global_params = {
        "TEST_SIZE": TEST_SIZE,
        "N_SPLITS": N_SPLITS
    }
    
    model_params = {
        "model_type": "GRU Neural Network",
        "gru_hyperparameters": {
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
        model_name="gru_diario",
        global_params=global_params,
        model_params=model_params,
        metrics=results,
        output_dir=OUTPUT_DIR / "reprodutibilidade"
    )


if __name__ == "__main__":
    main()
