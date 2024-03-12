import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
import joblib
from house_prices import CATEGORICAL_COLUMNS
from house_prices import MODEL_PATH
from house_prices import ENCODER_PATH
from house_prices import NUMERICAL_COLUMNS
from house_prices import SCALER_PATH


scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore')


def compute_rmsle(y_test: np.ndarray,
                  y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def scale_numerical_features(dataset, flag):
    if flag:
        scaler_set = scaler.fit(dataset[NUMERICAL_COLUMNS])
        joblib.dump(scaler_set, SCALER_PATH)
        scaler_set = scaler.transform(dataset[NUMERICAL_COLUMNS])
    else:
        scaler_set = scaler.transform(dataset[NUMERICAL_COLUMNS])
    return scaler_set


def encode_categorical_features(dataset, flag):
    if flag:
        encoded_set = encoder.fit(dataset[CATEGORICAL_COLUMNS])
        joblib.dump(encoded_set, ENCODER_PATH)
        encoded_set = encoder.transform(dataset[CATEGORICAL_COLUMNS]).toarray()
    else:
        encoded_set = encoder.transform(dataset[CATEGORICAL_COLUMNS]).toarray()
    return encoded_set


def preprocessor(dataset, flag):
    numerical_features = scale_numerical_features(dataset, flag)
    categorical_features = encode_categorical_features(dataset, flag)
    processed_feature = np.hstack([numerical_features, categorical_features])
    return processed_feature


def transform_target(y_train):
    y_train_transformed = np.log(y_train + 1)
    return y_train_transformed


def train_model(X_train_prepared, y_train_transformed):
    model = LinearRegression()
    model.fit(X_train_prepared, y_train_transformed)
    joblib.dump(model, MODEL_PATH)
    return model


def model_predict(model, X_test_prepared):
    y_pred_raw = model.predict(X_test_prepared)
    y_pred = np.exp(y_pred_raw) - 1
    return y_pred
