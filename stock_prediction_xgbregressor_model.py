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
