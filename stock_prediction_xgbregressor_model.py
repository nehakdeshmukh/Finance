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
