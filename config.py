from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'simulated_511_samples.csv'
ARTIFACT_DIR = ROOT / 'artifacts'
OUTPUT_DIR = ROOT / 'outputs'
FIGURE_DIR = ROOT / 'figures'

ID_COL = 'ID'
LABEL_COL = 'label'
FEATURES = [f'feature-{i}' for i in range(1, 8)]
CATEGORICAL_FEATURES = ['feature-2', 'feature-6']
CONTINUOUS_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
ORDINAL_FEATURES = ['feature-2', 'feature-6']
BINARY_FEATURES = []

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_GRID_POINTS = 50
N_BOOTSTRAP = 200
THRESHOLD_METHOD = 'youden'

ARTIFACT_FILES = {
    'rf_model': ARTIFACT_DIR / 'rf_model.pkl',
    'et_model': ARTIFACT_DIR / 'et_model.pkl',
    'catboost_model': ARTIFACT_DIR / 'catboost_model.pkl',
    'stacking_model': ARTIFACT_DIR / 'stacking_model.pkl',
    'preprocessor': ARTIFACT_DIR / 'preprocessor.pkl',
    'split_data': ARTIFACT_DIR / 'split_data.pkl',
    'schema': ARTIFACT_DIR / 'schema.json',
}

for path in [ARTIFACT_DIR, OUTPUT_DIR, FIGURE_DIR]:
    path.mkdir(exist_ok=True)
