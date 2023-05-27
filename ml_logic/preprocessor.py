import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df

    ohe = OneHotEncoder(sparse = False)
    ohe.fit(df[['payment_format']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['payment_format']])
    #ohe.fit(df[['currency_pair']])
    #df[ohe.get_feature_names_out()] = ohe.transform(df[['currency_pair']])


    print("✅ data preprocessed")

    return df
