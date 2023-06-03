import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df

    ohe = OneHotEncoder(sparse = False)
    ohe.fit(df[['payment_format']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['payment_format']])

    ohe.fit(df[['receiving_currency']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['receiving_currency']])

    ohe.fit(df[['payment_currency']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['payment_currency']])

    print("✅ data One-Hot-Encoded")

    return df
