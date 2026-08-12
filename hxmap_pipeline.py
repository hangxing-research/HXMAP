from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'simulated_511_samples.csv'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def load_data(path=DATA):
    df = pd.read_csv(path)
    features = [c for c in df.columns if c.startswith('feature-')]
    if not {'ID', 'label'}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'ID' and 'label' columns.")
    return df, features


def split_features(features):
    categorical = ['feature-2', 'feature-6']
    continuous = [f for f in features if f not in categorical]
    return continuous, categorical


def build_model(continuous, categorical):
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), continuous),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])
    return Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=2000, random_state=42)),
    ])


def evaluate(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return pd.DataFrame({
        'metric': ['auc', 'accuracy', 'f1', 'brier', 'n'],
        'value': [
            roc_auc_score(y_true, prob),
            accuracy_score(y_true, pred),
            f1_score(y_true, pred),
            brier_score_loss(y_true, prob),
            len(y_true),
        ],
    })


def plot_importance(imp_df, out_file):
    plot_df = imp_df.sort_values('importance', ascending=True)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.barh(plot_df['feature'], plot_df['importance'])
    plt.xlabel('Permutation Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    plt.close()


def main():
    df, features = load_data()
    continuous, categorical = split_features(features)
    X = df[features].copy()
    y = df['label'].astype(int).copy()
    ids = df['ID'].copy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.30, random_state=42, stratify=y
    )

    pipe = build_model(continuous, categorical)
    pipe.fit(X_train, y_train)

    train_prob = pipe.predict_proba(X_train)[:, 1]
    test_prob = pipe.predict_proba(X_test)[:, 1]

    train_metrics = evaluate(y_train, train_prob)
    train_metrics.insert(0, 'split', 'train')
    test_metrics = evaluate(y_test, test_prob)
    test_metrics.insert(0, 'split', 'test')
    metrics = pd.concat([train_metrics, test_metrics], ignore_index=True)
    metrics.to_csv(OUT / 'metrics.csv', index=False)

    pred_df = pd.DataFrame({
        'ID': id_test.values,
        'observed_label': y_test.values,
        'predicted_probability': test_prob,
        'predicted_class_0.5': (test_prob >= 0.5).astype(int),
    }).sort_values('ID')
    pred_df.to_csv(OUT / 'test_predictions.csv', index=False)

    perm = permutation_importance(pipe, X_test, y_test, n_repeats=20, random_state=42, scoring='roc_auc')
    imp_df = pd.DataFrame({'feature': features, 'importance': perm.importances_mean}).sort_values('importance', ascending=False)
    imp_df.to_csv(OUT / 'feature_importance.csv', index=False)
    plot_importance(imp_df, FIG / 'feature_importance.png')

    summary = pd.DataFrame({
        'item': ['n_total', 'n_train', 'n_test', 'positive_rate_total', 'positive_rate_train', 'positive_rate_test'],
        'value': [len(df), len(X_train), len(X_test), y.mean(), y_train.mean(), y_test.mean()],
    })
    summary.to_csv(OUT / 'data_summary.csv', index=False)

    print('Saved:')
    for p in [OUT / 'metrics.csv', OUT / 'test_predictions.csv', OUT / 'feature_importance.csv', OUT / 'data_summary.csv', FIG / 'feature_importance.png']:
        print(p)


if __name__ == '__main__':
    main()
