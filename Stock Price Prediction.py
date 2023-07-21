# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:06:16 2023

@author: nehak
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go


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























