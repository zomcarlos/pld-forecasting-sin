import glob
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dados"

SUBMERCADOS = ["SUL", "SUDESTE", "NORDESTE", "NORTE"]

# Mapeamento de nomes nos CSVs de CARGA (formato diferente dos demais)
_CARGA_NAME_MAP = {
    "SUL": "Sul",
    "SUDESTE": "Sudeste/Centro-Oeste",
    "NORDESTE": "Nordeste",
    "NORTE": "Norte",
}


def load_pld(submercado: str) -> pd.Series:
    # PLD passou a ser horário em jan/2021 com a transição do modelo DECOMP para o DESSEM
    files = sorted(glob.glob(str(DATA_DIR / "PLD" / "pld_horario_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(
            f, sep=";", quotechar='"', encoding="utf-8",
            dtype={"MES_REFERENCIA": str, "DIA": str, "HORA": str},
        )
        df.columns = df.columns.str.strip().str.strip('"')
        df["SUBMERCADO"] = df["SUBMERCADO"].str.strip().str.strip('"')
        df["MES_REFERENCIA"] = df["MES_REFERENCIA"].str.strip().str.strip('"')
        df["DIA"] = df["DIA"].str.strip().str.strip('"')
        df = df[df["SUBMERCADO"] == submercado].copy()
        df["data"] = pd.to_datetime(
            df["MES_REFERENCIA"].str[:4] + "-"
            + df["MES_REFERENCIA"].str[4:6] + "-"
            + df["DIA"],
            format="%Y-%m-%d",
        )
        df["PLD_HORA"] = (
            df["PLD_HORA"].astype(str).str.replace(",", ".").astype(float)
        )
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    # Agregação para média diária, adotada como unidade experimental
    daily = full.groupby("data")["PLD_HORA"].mean().sort_index()
    # Garante índice temporal contínuo e regular exigido por modelos de séries temporais
    daily.index = pd.DatetimeIndex(daily.index, freq="D")
    daily.name = "pld_medio"
    return daily


def load_ena(submercado: str) -> pd.Series:
    files = sorted(glob.glob(str(DATA_DIR / "ENA" / "ENA_DIARIO_SUBSISTEMA_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df["nom_subsistema"] = df["nom_subsistema"].str.strip()
        df = df[df["nom_subsistema"] == submercado].copy()
        df["data"] = pd.to_datetime(df["ena_data"])
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    daily = full.set_index("data")["ena_armazenavel_regiao_mwmed"].sort_index()
    daily.index = pd.DatetimeIndex(daily.index, freq="D")
    daily.name = "ena"
    return daily


def load_ear(submercado: str) -> pd.Series:
    """Carrega EAR (Energia Armazenada) diário — nível do reservatório em %."""
    files = sorted(glob.glob(str(DATA_DIR / "EAR" / "EAR_DIARIO_SUBSISTEMA_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df["nom_subsistema"] = df["nom_subsistema"].str.strip()
        df = df[df["nom_subsistema"] == submercado].copy()
        df["data"] = pd.to_datetime(df["ear_data"])
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    daily = full.set_index("data")["ear_verif_subsistema_percentual"].sort_index()
    daily.index = pd.DatetimeIndex(daily.index, freq="D")
    daily.name = "ear_pct"
    return daily


def load_carga(submercado: str) -> pd.Series:
    """Carrega Carga (Demanda) diária em MWmed."""
    carga_name = _CARGA_NAME_MAP[submercado]
    files = sorted(glob.glob(str(DATA_DIR / "CARGA" / "CARGA_ENERGIA_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df["nom_subsistema"] = df["nom_subsistema"].str.strip()
        df = df[df["nom_subsistema"] == carga_name].copy()
        df["data"] = pd.to_datetime(df["din_instante"])
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    daily = full.set_index("data")["val_cargaenergiamwmed"].sort_index()
    daily.index = pd.DatetimeIndex(daily.index, freq="D")
    daily.name = "carga"
    return daily


def get_aligned_data(submercado: str) -> pd.DataFrame:
    """Retorna DataFrame alinhado com PLD + todas as exógenas."""
    pld = load_pld(submercado)
    ena = load_ena(submercado)
    ear = load_ear(submercado)
    carga = load_carga(submercado)

    # Interseção garante alinhamento temporal entre variáveis com diferentes coberturas de datas
    common_idx = pld.index.intersection(ena.index).intersection(ear.index).intersection(carga.index)

    df = pd.DataFrame({
        "pld": pld.loc[common_idx],
        "ena": ena.loc[common_idx],
        "ear_pct": ear.loc[common_idx],
        "carga": carga.loc[common_idx],
    })
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering para previsão diária de PLD."""
    feat_df = df.copy()

    # --- Lags autoregressivos do PLD ---
    # Lags até 14 dias capturam a sazonalidade semanal e quinzenal do PLD
    for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
        feat_df[f"pld_lag_{lag}"] = feat_df["pld"].shift(lag)

    # --- Estatísticas rolantes do PLD ---
    # shift(1) antes da janela móvel evita vazamento de dados (data leakage) na predição
    for window in [7, 14, 30]:
        feat_df[f"pld_rolling_mean_{window}"] = feat_df["pld"].shift(1).rolling(window).mean()
        feat_df[f"pld_rolling_std_{window}"] = feat_df["pld"].shift(1).rolling(window).std()

    feat_df["pld_rolling_min_7"] = feat_df["pld"].shift(1).rolling(7).min()
    feat_df["pld_rolling_max_7"] = feat_df["pld"].shift(1).rolling(7).max()

    # --- Momentum ---
    # Primeira diferença defasada para capturar a direção da tendência recente
    feat_df["pld_diff_1"] = feat_df["pld"].shift(1) - feat_df["pld"].shift(2)

    # --- ENA features ---
    feat_df["ena_lag_1"] = feat_df["ena"].shift(1)
    feat_df["ena_rolling_mean_7"] = feat_df["ena"].shift(1).rolling(7).mean()
    feat_df["ena_rolling_mean_30"] = feat_df["ena"].shift(1).rolling(30).mean()

    # --- EAR features (nível do reservatório) ---
    feat_df["ear_lag_1"] = feat_df["ear_pct"].shift(1)
    feat_df["ear_rolling_mean_7"] = feat_df["ear_pct"].shift(1).rolling(7).mean()
    feat_df["ear_rolling_mean_30"] = feat_df["ear_pct"].shift(1).rolling(30).mean()
    # Variação do nível do reservatório (enchendo ou esvaziando?)
    # Trajetória de recarga/depleção é determinante para a expectativa de formação do preço
    feat_df["ear_diff_1"] = feat_df["ear_pct"].shift(1) - feat_df["ear_pct"].shift(2)
    feat_df["ear_diff_7"] = feat_df["ear_pct"].shift(1) - feat_df["ear_pct"].shift(8)

    # --- CARGA features (demanda) ---
    feat_df["carga_lag_1"] = feat_df["carga"].shift(1)
    feat_df["carga_rolling_mean_7"] = feat_df["carga"].shift(1).rolling(7).mean()

    # --- Features calendário ---
    # Capturam sazonalidade e padrão de menor demanda/preço aos finais de semana
    feat_df["day_of_week"] = feat_df.index.dayofweek
    feat_df["month"] = feat_df.index.month
    feat_df["day_of_year"] = feat_df.index.dayofyear
    feat_df["is_weekend"] = (feat_df.index.dayofweek >= 5).astype(int)

    return feat_df
