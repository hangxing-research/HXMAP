import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import expit
from sklearn.inspection import permutation_importance
from .config import FEATURES, CONTINUOUS_FEATURES, N_GRID_POINTS


def get_feature_importance(base_models, X_train, y_train):
    rows = []
    for name, pipe in base_models.items():
        perm = permutation_importance(
            pipe,
            X_train[FEATURES],
            y_train,
            n_repeats=20,
            random_state=42,
            scoring='roc_auc',
        )
        imp = np.asarray(perm.importances_mean, dtype=float)
        imp = np.clip(imp, 0, None)
        if imp.sum() <= 0:
            imp = np.repeat(1 / len(FEATURES), len(FEATURES))
        else:
            imp = imp / imp.sum()
        rows.append(pd.DataFrame({'Feature': FEATURES, f'{name} Importance': imp}))
    out = rows[0]
    for df in rows[1:]:
        out = out.merge(df, on='Feature')
    return out


def get_meta_coefficients(stacking_model):
    final_estimator = stacking_model.final_estimator_
    coefs = final_estimator.coef_[0]
    intercept = float(final_estimator.intercept_[0])
    names = list(stacking_model.named_estimators_.keys())
    return dict(zip(names, coefs)), intercept


def create_feature_grid(feature, training_data, n_points=N_GRID_POINTS):
    values = training_data[feature].dropna()
    if feature in CONTINUOUS_FEATURES:
        return np.linspace(values.quantile(0.01), values.quantile(0.99), n_points)
    return np.sort(values.unique()).astype(float)


def get_reference_values(training_data, binary_features=None):
    binary_features = binary_features or []
    refs = {}
    for feature in FEATURES:
        refs[feature] = training_data[feature].mode().iloc[0] if feature in binary_features else training_data[feature].median()
    return refs


def average_model_probability(model, feature, feature_value, X_background_raw):
    X_modified = X_background_raw[FEATURES].copy()
    X_modified[feature] = feature_value
    return model.predict_proba(X_modified)[:, 1].mean()


def build_fma_mapping(base_models, feature_importances_df, meta_coeffs, X_train_raw, reference_values):
    importance_name_map = {name: f'{name} Importance' for name in base_models.keys()}
    rows = []
    for feature in FEATURES:
        feature_grid = create_feature_grid(feature, X_train_raw)
        reference_value = reference_values[feature]
        feature_row = feature_importances_df.set_index('Feature').loc[feature]
        reference_probabilities = {name: average_model_probability(model, feature, reference_value, X_train_raw) for name, model in base_models.items()}
        for feature_value in feature_grid:
            total_delta_cf = 0.0
            row = {'Feature': feature, 'Feature value': float(feature_value), 'Reference value': float(reference_value)}
            for model_name, model in base_models.items():
                model_probability = average_model_probability(model, feature, feature_value, X_train_raw)
                delta_probability = model_probability - reference_probabilities[model_name]
                feature_importance = float(feature_row[importance_name_map[model_name]])
                meta_coefficient = float(meta_coeffs[model_name])
                model_delta_cf = feature_importance * meta_coefficient * delta_probability
                total_delta_cf += model_delta_cf
                row[f'{model_name} delta CF'] = model_delta_cf
            row['Delta CF'] = total_delta_cf
            rows.append(row)
    mapping = pd.DataFrame(rows)
    ranges = mapping.groupby('Feature')['Delta CF'].agg(['min', 'max'])
    ranges['Contribution range'] = ranges['max'] - ranges['min']
    largest_range = float(ranges['Contribution range'].max())
    minimum_cf_by_feature = mapping.groupby('Feature')['Delta CF'].min().to_dict()
    mapping['Nomogram Points'] = mapping.apply(lambda r: (r['Delta CF'] - minimum_cf_by_feature[r['Feature']]) / largest_range * 100 if largest_range > 0 else 0.0, axis=1)
    return mapping, minimum_cf_by_feature, largest_range


def build_interpolators(mapping_df):
    cf_funcs, point_funcs = {}, {}
    for feature in FEATURES:
        df = mapping_df[mapping_df['Feature'] == feature].sort_values('Feature value')
        x = df['Feature value'].to_numpy(dtype=float)
        cf = df['Delta CF'].to_numpy(dtype=float)
        pts = df['Nomogram Points'].to_numpy(dtype=float)
        cf_funcs[feature] = interp1d(x, cf, kind='linear', bounds_error=False, fill_value=(cf[0], cf[-1]))
        point_funcs[feature] = interp1d(x, pts, kind='linear', bounds_error=False, fill_value=(pts[0], pts[-1]))
    return cf_funcs, point_funcs


def score_patient(row, feature_order, mapping_functions):
    total = 0.0
    details = {}
    for feature in feature_order:
        val = float(row[feature])
        score = float(mapping_functions[feature](val))
        details[feature] = score
        total += score
    return total, details


def compute_total_cf(X_raw, cf_functions):
    return np.asarray([score_patient(row, FEATURES, cf_functions)[0] for _, row in X_raw.iterrows()])


def fit_total_cf_to_stacking_logit(total_cf_train, stacking_train_prob):
    p = np.clip(np.asarray(stacking_train_prob, dtype=float), 1e-6, 1 - 1e-6)
    target_logit = np.log(p / (1 - p))
    slope, intercept = np.polyfit(np.asarray(total_cf_train, dtype=float), target_logit, 1)
    return float(intercept), float(slope)


def map_total_cf_to_probability(total_cf, lp_intercept, lp_slope):
    lp = lp_intercept + lp_slope * np.asarray(total_cf, dtype=float)
    prob = expit(lp)
    return lp, prob
