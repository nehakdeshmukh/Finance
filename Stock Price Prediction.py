# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:06:16 2023

@author: nehak
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.preprocessing

# import all stock prices 
data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\prices-split-adjusted.csv", index_col = 0)

data.head(5)

data.info()

print("unique shares:", len(data["symbol"].value_counts()))

data.describe()


# AAPL stock 
fig = go.Figure()
fig.add_trace(go.Scatter(y= data[data.symbol == 'AAPL'].open.values,name='open'),)
fig.add_trace(go.Scatter(y= data[data.symbol == 'AAPL'].close.values,name='close'))
fig.add_trace(go.Scatter(y= data[data.symbol == 'AAPL'].low.values,name='low'))
fig.add_trace(go.Scatter(y= data[data.symbol == 'AAPL'].high.values,name='high'))
fig.update_layout(title="Stock Price",xaxis_title="time in days",  yaxis_title="Price")
fig.show()


fig = go.Figure()

fig.add_trace(go.Scatter(y= data[data.symbol == 'AAPL'].volume.values,name='volume'),)
fig.update_layout(title="stock volume",xaxis_title="time in days",  yaxis_title="volume")
fig.show()

# Normalization 

# function for min-max normalization of stock
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df

# choose one stock
df_stock = data[data.symbol == 'AAPL'].copy()
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)












