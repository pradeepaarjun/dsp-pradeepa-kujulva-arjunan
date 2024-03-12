import numpy as np
import pandas as pd
import joblib
from house_prices import CATEGORICAL_COLUMNS
from house_prices import NUMERICAL_COLUMNS
from house_prices import MODEL_PATH
from house_prices.preprocess import scale_numerical_features
from house_prices.preprocess import encode_categorical_features


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    feature = input_data[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
    numerical_features = scale_numerical_features(feature, False)
    categorical_features = encode_categorical_features(feature, False)
    processed_test_feature = np.hstack([numerical_features,
                                        categorical_features])
    acutal_prediction = joblib.load(MODEL_PATH).predict(processed_test_feature)
    submission_df = pd.DataFrame({'Id': input_data['Id'],
                                  'SalePrice': np.exp(acutal_prediction)})
    return submission_df
