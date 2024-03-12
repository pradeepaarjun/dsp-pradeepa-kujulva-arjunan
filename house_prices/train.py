import pandas as pd
from sklearn.model_selection import train_test_split
from house_prices import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMNS
from house_prices.preprocess import preprocessor
from house_prices.preprocess import model_predict
from house_prices.preprocess import train_model
from house_prices.preprocess import transform_target
from house_prices.preprocess import compute_rmsle


def build_model(data: pd.DataFrame) -> dict[str, str]:
    feature, target = data[NUMERICAL_COLUMNS +
                           CATEGORICAL_COLUMNS], data[TARGET_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(
        feature, target, test_size=0.2, random_state=42)
    processed_feature_train = preprocessor(X_train, True)
    processed_feature_test = preprocessor(X_test, True)
    y_train_transformed = transform_target(y_train)
    model = train_model(processed_feature_train, y_train_transformed)
    y_pred = model_predict(model, processed_feature_test)
    rmsle_score = compute_rmsle(y_test, y_pred)
    result = {'RMSLE': str(rmsle_score)}
    return result
