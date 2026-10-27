import pandas as pd
from .config import TARGET

def load(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Нет целевой колонки {TARGET!r}")
    dups = df.duplicated().sum()
    if dups:
        print(f"[data] удалено дубликатов: {dups}")
        df = df.drop_duplicates()
    return df.drop(columns=TARGET), df[TARGET]

def eda(X, y):
    print(f"[eda] строк={len(X)} признаков={X.shape[1]}")
    print(f"[eda] пропусков={int(X.isna().sum().sum())}")
    share = y.value_counts(normalize=True).round(3).to_dict()
    print(f"[eda] баланс классов: {share}")
    top = X.corrwith(y).abs().sort_values(ascending=False).head(3)
    print(f"[eda] топ-3 |corr|: {top.round(2).to_dict()}")
