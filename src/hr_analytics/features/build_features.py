# src/hr_analytics/features/build_features.py

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from hr_analytics.data.preprocess import build_imputer


def build_preprocessor(num_cols, ord_cols, ohe_cols):

    ohe_pipe = Pipeline([
        ("imputer", build_imputer()),
        ("ohe", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    ord_pipe = Pipeline([
        ("ord", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan
        ))
    ])

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("ord", ord_pipe, ord_cols),
        ("ohe", ohe_pipe, ohe_cols)
    ])