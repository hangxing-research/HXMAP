import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, FEATURES, ID_COL, LABEL_COL, TEST_SIZE, RANDOM_STATE


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {ID_COL, LABEL_COL, *FEATURES}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return df


def make_split(df):
    X = df[FEATURES].copy()
    y = df[LABEL_COL].astype(int).copy()
    ids = df[ID_COL].copy()
    return train_test_split(X, y, ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
