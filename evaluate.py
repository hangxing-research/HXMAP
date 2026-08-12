import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.linear_model import LogisticRegression
from .config import N_BOOTSTRAP


def youden_threshold(y_true, prob):
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def classification_metrics(y_true, prob, threshold):
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        'AUC': roc_auc_score(y_true, prob),
        'Accuracy': accuracy_score(y_true, pred),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1-score': f1_score(y_true, pred, zero_division=0),
        'Brier score': brier_score_loss(y_true, prob),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }


def calibration_intercept_slope(y_true, prob):
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        try:
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        except Exception:
            model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000)
        model.fit(x, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def bootstrap_performance(y_true, prob, threshold, n_iterations=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    metrics = []
    for _ in range(n_iterations):
        idx = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        yb, pb = y_true[idx], prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = classification_metrics(yb, pb, threshold)
        try:
            ci, cs = calibration_intercept_slope(yb, pb)
        except Exception:
            ci, cs = np.nan, np.nan
        m['Calibration intercept'] = ci
        m['Calibration slope'] = cs
        metrics.append(m)
    return pd.DataFrame(metrics)
