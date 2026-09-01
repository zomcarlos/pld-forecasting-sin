import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from data_loader import get_aligned_data, build_features

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "output" / "sensitivity.csv"

def compute_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_sensitivity(submercado="SUDESTE"):
    df = get_aligned_data(submercado)
    feat_df = build_features(df).dropna()
    
    y = np.log(feat_df["pld"])
    X = feat_df.drop(columns=["pld"])
    
    tscv = TimeSeriesSplit(n_splits=4, test_size=30)
    results = []

    print(f"Iniciando análise de sensibilidade no submercado {submercado}...")

    # 1. Ridge
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    for alpha in alphas:
        mae_list, rmse_list, mape_list = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_s, y_train)
            
            y_pred = np.exp(model.predict(X_test_s))
            y_true = np.exp(y_test.values)
            
            mae_list.append(mean_absolute_error(y_true, y_pred))
            rmse_list.append(root_mean_squared_error(y_true, y_pred))
            mape_list.append(compute_mape(y_true, y_pred))
            
        results.append({
            "modelo": "ridge",
            "hiperparametro": "alpha",
            "valor": alpha,
            "mae": np.mean(mae_list),
            "rmse": np.mean(rmse_list),
            "mape": np.mean(mape_list)
        })
        
    # 2. Random Forest
    n_estimators_list = [50, 100, 200, 500]
    for n_est in n_estimators_list:
        mae_list, rmse_list, mape_list = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = np.exp(model.predict(X_test))
            y_true = np.exp(y_test.values)
            
            mae_list.append(mean_absolute_error(y_true, y_pred))
            rmse_list.append(root_mean_squared_error(y_true, y_pred))
            mape_list.append(compute_mape(y_true, y_pred))
            
        results.append({
            "modelo": "rf",
            "hiperparametro": "n_estimators",
            "valor": n_est,
            "mae": np.mean(mae_list),
            "rmse": np.mean(rmse_list),
            "mape": np.mean(mape_list)
        })

    # 3. LightGBM
    learning_rates = [0.01, 0.05, 0.1]
    for lr in learning_rates:
        mae_list, rmse_list, mape_list = [], [], []
        for train_idx, test_idx in tscv.split(X):
            split_point = int(len(train_idx) * 0.9)
            train_sub_idx = train_idx[:split_point]
            val_sub_idx = train_idx[split_point:]
            
            X_train_sub = X.iloc[train_sub_idx]
            y_train_sub = y.iloc[train_sub_idx]
            X_val = X.iloc[val_sub_idx]
            y_val = y.iloc[val_sub_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'n_estimators': 500,
                'learning_rate': lr,
                'num_leaves': 31,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_sub, y_train_sub,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
                      
            y_pred = np.exp(model.predict(X_test))
            y_true = np.exp(y_test.values)
            
            mae_list.append(mean_absolute_error(y_true, y_pred))
            rmse_list.append(root_mean_squared_error(y_true, y_pred))
            mape_list.append(compute_mape(y_true, y_pred))
            
        results.append({
            "modelo": "lightgbm",
            "hiperparametro": "learning_rate",
            "valor": lr,
            "mae": np.mean(mae_list),
            "rmse": np.mean(rmse_list),
            "mape": np.mean(mape_list)
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, sep=";", index=False)
    print(f"Análise de sensibilidade concluída e salva em {OUTPUT_FILE}")

if __name__ == "__main__":
    run_sensitivity()
