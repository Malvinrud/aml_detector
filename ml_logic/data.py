import pandas as pd
import numpy as np
import csv
from pathlib import Path
import requests

# from taxifare.params import *

def get_data_local(size="Small", fraud="HI"):

    """Enter "Small"/"Medium"/"Large" to specify corpus size. "HI"/"LI" for amount of fraud inside (HI is more)."""

    fraud = fraud

    size = size

    file = f'../raw_data/{fraud}-{size}_Trans.csv'
    df = pd.read_csv(file, decimal=',')

    return df

def get_data_cloud():

    pass

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning and necessary preprocessing task.
    """


    # List of payment formats to remove
    remove_formats = ['Reinvestment', 'Wire']

    # Remove the rows with the specified payment formats
    df = df[~df['Payment Format'].isin(remove_formats)]

    df = df.rename(columns={"Timestamp": "timestamp",
                   "From Bank": "from_bank",
                   "Account": "from_account",
                   "Account.1": "to_account",
                   "From Bank": "from_bank",
                   "To Bank": "to_bank",
                   "Amount Received": "amount_received",
                   "Amount Paid": "amount_paid",
                   "Payment Format": "payment_format",
                   "Is Laundering": "is_laundering"})

    currency_dict = {"US Dollar": "USD",
                 "Bitcoin": "XBT",
                 "Euro": "EUR",
                 "Australiean Dollar": "AUD",
                 "Yuan": "CNY",
                 "Rupee": "INR",
                 "Yen": "JPY",
                 "Mexican Peso": "MXN",
                 "UK Pound": "GBP",
                 "Ruble": "RUB",
                 "Canadian Dollar": "CAD",
                 "Swiss Franc": "CHF",
                 "Brazil Real": "BRL",
                 "Saudi Riyal": "SAR",
                 "Shekel": "ILS"}

    df['currency_code'] = df['Payment Currency'].replace(currency_dict)

    # Get the exchange rate for all the currencies by connecting to Exchange Rate Data API

    # Exchange rate data api key
    api_key = "wdiQOxfDckJMZr70O1Brmvlh56iJEfE7"

    # Arbitrary date to fetch the exchange rates
    date = "2022-09-30"

    # The list of currency codes to fetch the exchange rates
    currency_codes = ["GBP", "EUR", "AUD", "BTC", "BRL", "CAD", "MXN", "RUB", "INR", "SAR", "ILS", "CHF", "JPY", "CNY"]

    # URL
    url = f"https://api.apilayer.com/exchangerates_data/{date}?symbols={','.join(currency_codes)}&base=USD"

    # Define the headers
    headers = {
    "apikey": api_key
    }

    # Send a GET request to the API
    response = requests.get(url, headers=headers)

    # Convert the response to JSON
    data = response.json()

    # Create a DataFrame from the rates
    df_rates = pd.DataFrame(data['rates'].items(), columns=['currency_code', 'rate'])

    # Merge df with df_rates on 'Currency Code', preserving all rows from df and filling in NaN for missing match
    df = df.merge(df_rates, on='currency_code', how='left')

    # Wherever the rate is NaN, that means the currency was USD. We can fill those with 1.
    df['rate'] = df['rate'].fillna(1)

    # Compute 'Amount Paid USD' and 'Amount Received USD'
    df['amount_paid_USD'] = df['amount_paid'].astype('float32') * df['rate']
    df['amount_received_USD'] = df['amount_received'].astype('float32') * df['rate']


    # put currency pair together and delete obsolete columns

    df["Receiving Currency"] = df["Receiving Currency"].map(currency_dict)
    df["Payment Currency"] = df["Payment Currency"].map(currency_dict)

    df["currency_pair"] = df["Receiving Currency"] + "_" + df["Payment Currency"]
    df = df.drop(['Receiving Currency', 'Payment Currency'], axis=1)


    # Convert Timestamp into datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # Create new columns for year, month, day, hour and minute
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    df.drop("amount_received", axis=1, inplace=True)
    df.drop("amount_paid", axis=1, inplace=True)
    df.drop("currency_code", axis=1, inplace=True)
    df.drop("rate", axis=1, inplace=True)


    print("✅ data cleaned")

    return df
