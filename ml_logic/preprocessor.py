import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df

    lst=['Cheque', 'Credit Card', 'ACH', 'Cash', 'Bitcoin', 'Cheque', 'Credit Card', 'ACH', 'Cash', 'Bitcoin', 'Cheque', 'Credit Card', 'ACH', 'Cash', 'Bitcoin']

    lst2=["USD", "XBT", "EUR", "AUD", "CNY", "INR", "JPY", "MXN", "GBP", "RUB", "CAD", "CHF", "BRL", "SAR", "ILS"]

    lst3=["USD", "XBT", "EUR", "AUD", "CNY", "INR", "JPY", "MXN", "GBP", "RUB", "CAD", "CHF", "BRL", "SAR", "ILS"]


    data = pd.DataFrame(list(zip(lst, lst2, lst3)), columns =['payment_format', 'receiving_currency', 'payment_currency'])

    ohe = OneHotEncoder(sparse = False, handle_unknown="ignore")
    ohe.fit(data[['payment_format']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['payment_format']])

    ohe.fit(data[['receiving_currency']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['receiving_currency']])

    ohe.fit(data[['payment_currency']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['payment_currency']])

    print("✅ data One-Hot-Encoded")

    return df
