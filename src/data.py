"""
Data loading utilities.
"""

import pandas as pd
from config import Config


def load_ethereum_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    df.columns = df.columns.str.strip()
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    df = df.drop(columns=cat_cols)
    df = df.fillna(0)
    df.insert(0, "addr_id", [f"ADDR_{i:05d}" for i in range(len(df))])
    return df
