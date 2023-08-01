# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:12:12 2023

@author: nehak
"""

import numpy as np
import pandas as pd

df_technical = pd.read_csv(
    r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\prices-split-adjusted.csv")
df_fundamental = pd.read_csv(
    r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\fundamentals.csv")
df_fundamental = df_fundamental.drop("Unnamed: 0", axis=1)
df_securities = pd.read_csv(
    r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\securities.csv")


print(df_technical)
print(df_technical.info())
df_technical.describe()

print(df_fundamental)
print(df_fundamental.info())
df_fundamental.describe()

print(df_securities)
print(df_securities.info())
df_securities.describe()


df_securities["GICS Sector"].value_counts()

# Filter IT sector stocks

all_tech_symbol = list(df_securities[df_securities["GICS Sector"]=="Information Technology"]["Ticker symbol"])

df_technical_tech = df_technical[df_technical["symbol"].isin(all_tech_symbol)]
symbol_close = df_technical_tech.groupby("symbol")["volume"].mean().reset_index()
symbol_close = symbol_close.sort_values(by="volume", ascending=False).reset_index(drop=True)[:3]
top_tech_symbol = list(symbol_close["symbol"])

print(df_securities[df_securities["Ticker symbol"].isin(top_tech_symbol)].reset_index(drop=True))

df_technical = df_technical[df_technical["symbol"].isin(top_tech_symbol)].reset_index(drop=True)
df_fundamental = df_fundamental[df_fundamental["Ticker Symbol"].isin(top_tech_symbol)].reset_index(drop=True)

print(df_technical)
print(df_fundamental)

# Datetime Features
print("Before Adding Datetime Features:", df_technical.shape)

col = "date"
prefix = col + "_"
df_technical[col] = pd.to_datetime(df_technical[col])
df_technical[prefix + 'year'] = df_technical[col].dt.year
df_technical[prefix + 'month'] = df_technical[col].dt.month
df_technical[prefix + 'day'] = df_technical[col].dt.day
df_technical[prefix + 'weekofyear'] = df_technical[col].dt.weekofyear
df_technical[prefix + 'dayofweek'] = df_technical[col].dt.dayofweek
df_technical[prefix + 'quarter'] = df_technical[col].dt.quarter
df_technical[prefix + 'is_month_start'] = df_technical[col].dt.is_month_start.astype(int)
df_technical[prefix + 'is_month_end'] = df_technical[col].dt.is_month_end.astype(int)

print(" After Adding Datetime Features:", df_technical.shape)

df_technical

# Lag Features
print("Before Adding Lag Features:", df_technical.shape)

ohlcv = ["open", "high", "low", "close", "volume"]
for col in ohlcv:
    lags = np.arange(65, 101, 1)
    for lag in lags:
        
        df_technical["{}_lag_{}".format(col, lag)] = df_technical.groupby("symbol")[col].transform(lambda x: x.shift(lag))

print(" After Adding Lag Features:", df_technical.shape)

df_technical

# Drop Features
print("Before Drop Features:", df_technical.shape)

drop_features = ["open", "high", "low", "volume"]
df_technical = df_technical.drop(drop_features, axis=1)

print(" After Drop Features:", df_technical.shape)

df_technical

# Datetime Features Fundamental features
print("Before Adding Datetime Features:", df_fundamental.shape)

df_fundamental["Period Ending"] = pd.to_datetime(df_fundamental["Period Ending"])
df_fundamental["First_Date_Tech"] = df_fundamental["Period Ending"] + pd.Timedelta(days=1)
df_fundamental['Last_Date_Tech'] = df_fundamental.groupby("Ticker Symbol")["Period Ending"].shift(-1)
df_fundamental['Last_Date_Tech'] = df_fundamental['Last_Date_Tech'].fillna(df_fundamental['Period Ending'] + pd.offsets.DateOffset(years=1))

print(" After Adding Datetime Features:", df_fundamental.shape)

df_fundamental

# Drop Features
print("Before Drop Features:", df_fundamental.shape)

# drop_features = ["Period Ending", "Cash Ratio", "Current Ratio", "Quick Ratio", "For Year", "Earnings Per Share", "Estimated Shares Outstanding"]
drop_features = ["Period Ending", "For Year"]
df_fundamental = df_fundamental.drop(drop_features, axis=1)

print(" After Drop Features:", df_fundamental.shape)

df_fundamental

# Filtering Date and Symbol in Technical Data
df_technical = df_technical[(df_technical["date"]>="2014-01-01")&(df_technical["date"]<="2016-12-31")].reset_index(drop=True)
# df_technical = df_technical[(df_technical["date"]>="2010-01-01")&(df_technical["date"]<="2016-12-31")].reset_index(drop=True)
df_technical

# Merging Technical and Fundamental Data
df_tech_fund = df_technical.merge(df_fundamental, left_on="symbol", right_on="Ticker Symbol")
df_tech_fund = df_tech_fund[(df_tech_fund["date"]>=df_tech_fund["First_Date_Tech"]) & (df_tech_fund["date"]<=df_tech_fund["Last_Date_Tech"])].reset_index(drop=True)

# df_tech_fund = df_technical.copy()
df_tech_fund


# Drop Features
print("Before Drop Features:", df_tech_fund.shape)

drop_features = ["Ticker Symbol", "First_Date_Tech", "Last_Date_Tech"]
df_tech_fund = df_tech_fund.drop(drop_features, axis=1)

print(" After Drop Features:", df_tech_fund.shape)

df_tech_fund

# Data Splitting (Train: 2014-Q3 2016, Test: Q4 2016)
df_train = df_tech_fund[df_tech_fund["date"] < "2016-10-01"].reset_index(drop=True)
df_test = df_tech_fund[df_tech_fund["date"] >= "2016-10-01"].reset_index(drop=True)

# Create Actual Predicted DataFrame
df_actual_predicted = pd.DataFrame({
    "date": df_train["date"],
    "symbol": df_train["symbol"],
    "actual": df_train["close"], 
})

df_actual_predicted_test = pd.DataFrame({
    "date": df_test["date"],
    "symbol": df_test["symbol"],
    "actual": df_test["close"], 
})

df_actual_predicted = df_actual_predicted.append(df_actual_predicted_test, ignore_index=True)
