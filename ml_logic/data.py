import pandas as pd
import numpy as np
import csv
import requests
import os

# from taxifare.params import *

def get_data_local(size="Small", fraud="HI"):

    """Enter "Small"/"Medium"/"Large" to specify corpus size. "HI"/"LI" for amount of fraud inside (HI is more)."""

    fraud = fraud

    size = size

    current_dir = os.path.dirname(__file__)
    file = os.path.join(current_dir, '..', 'raw_data', fraud+"-"+size+"_Trans.csv")

    #file = f'../raw_data/{fraud}-{size}_Trans.csv'
    df = pd.read_csv(file, decimal=',')

    print("✅ data received")

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
                   "Is Laundering": "is_laundering",
                   "Receiving Currency": "receiving_currency",
                   "Payment Currency": "payment_currency"})

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

    df['currency_code'] = df['payment_currency'].replace(currency_dict)

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

    df["receiving_currency"] = df["receiving_currency"].map(currency_dict)
    df["payment_currency"] = df["payment_currency"].map(currency_dict)

    # df["currency_pair"] = df["receiving_currency"] + "_" + df["payment_currency"]
    # df = df.drop(['receiving_currency', 'payment_currency'], axis=1)


    # Convert Timestamp into datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # Create new columns for year, month, day, hour and minute
    # df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    df.drop("amount_received", axis=1, inplace=True)
    df.drop("amount_paid", axis=1, inplace=True)
    df.drop("currency_code", axis=1, inplace=True)
    df.drop("rate", axis=1, inplace=True)
    df.drop("timestamp", axis=1, inplace=True)


    print("✅ data cleaned")

    return df


def data_reduction(df, expected_fraud_amount=0.1):
    """
    This function randomly cuts non-fraud transactions to artificially increase amount of fraud
    """

    # list all unique accounts

    from_accounts = df["from_account"].to_numpy()
    to_accounts  = df["to_account"].to_numpy()

    all_accounts = np.append(from_accounts, to_accounts)
    all_accounts = np.unique(all_accounts)

    # show only fraud

    fraud_mask = df["is_laundering"] == 1

    fraud_df = df[fraud_mask]

    fraud_from = fraud_df["from_account"].to_numpy()
    fraud_to  = fraud_df["to_account"].to_numpy()

    all_fraud = np.append(fraud_from, fraud_to)
    all_fraud = np.unique(all_fraud)

    # keep this part for final df

    all_fraud_df_mask = df["from_account"].isin(all_fraud) | df["to_account"].isin(all_fraud)

    all_fraud_df = df[all_fraud_df_mask]

    # get df with only non_fraud data

    non_fraud_mask = ~(df["from_account"].isin(all_fraud) | df["to_account"].isin(all_fraud))

    non_fraud_df = df[non_fraud_mask]


    # getting amount for non-fraud

    all_fraud_absolut = len(fraud_df)

    new_amount_total = all_fraud_absolut/expected_fraud_amount

    new_amount_non_fraud = int(new_amount_total - all_fraud_absolut)

    new_non_fraud_df = non_fraud_df.sample(n=new_amount_non_fraud)

    new_df = pd.concat([new_non_fraud_df, fraud_df])

    print(f"✅ randomly removed {len(df)-len(new_df)} non-fraudulent rows from the data frame.")

    return new_df
