import os


NUMERICAL_COLUMNS = ['LotArea', 'GrLivArea']
CATEGORICAL_COLUMNS = ['MSZoning', 'Street']
TARGET_COLUMNS = 'SalePrice'
MODEL_PATH = os.path.join('..', 'models', 'model.joblib')
ENCODER_PATH = os.path.join('..', 'models', 'encoder.joblib')
SCALER_PATH = os.path.join('..', 'models', 'scaler.joblib')
