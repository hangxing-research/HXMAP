import json
import joblib
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, ARTIFACT_FILES, FEATURES, RANDOM_STATE
from .data import load_dataset, make_split


def build_preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), CONTINUOUS_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL_FEATURES),
    ])


def build_base_pipelines():
    rf = Pipeline([('prep', build_preprocessor()), ('model', RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))])
    et = Pipeline([('prep', build_preprocessor()), ('model', ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))])
    cat = Pipeline([('prep', build_preprocessor()), ('model', CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, allow_writing_files=False))])
    return rf, et, cat


def fit_and_save_artifacts():
    df = load_dataset()
    X_train, X_test, y_train, y_test, id_train, id_test = make_split(df)
    rf, et, cat = build_base_pipelines()
    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    estimators = [('rf', rf), ('et', et), ('catboost', cat)]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method='predict_proba',
        passthrough=False,
        cv=5,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    joblib.dump(rf, ARTIFACT_FILES['rf_model'])
    joblib.dump(et, ARTIFACT_FILES['et_model'])
    joblib.dump(cat, ARTIFACT_FILES['catboost_model'])
    joblib.dump(stacking, ARTIFACT_FILES['stacking_model'])
    joblib.dump(build_preprocessor().fit(X_train), ARTIFACT_FILES['preprocessor'])
    joblib.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'id_train': id_train, 'id_test': id_test}, ARTIFACT_FILES['split_data'])

    schema = {
        'features': FEATURES,
        'continuous_features': CONTINUOUS_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'base_models': ['rf', 'et', 'catboost'],
        'meta_model': 'logistic_regression',
    }
    ARTIFACT_FILES['schema'].write_text(json.dumps(schema, indent=2), encoding='utf-8')


def load_artifacts():
    return {
        'rf_model': joblib.load(ARTIFACT_FILES['rf_model']),
        'et_model': joblib.load(ARTIFACT_FILES['et_model']),
        'catboost_model': joblib.load(ARTIFACT_FILES['catboost_model']),
        'stacking_model': joblib.load(ARTIFACT_FILES['stacking_model']),
        'preprocessor': joblib.load(ARTIFACT_FILES['preprocessor']),
        'split_data': joblib.load(ARTIFACT_FILES['split_data']),
    }


def artifacts_exist():
    return all(path.exists() for key, path in ARTIFACT_FILES.items() if key != 'schema')
