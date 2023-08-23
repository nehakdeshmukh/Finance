# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:41:10 2023

@author: nehak
"""


# Shares Prediction using LSTM
import pandas
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np 

yf.pdr_override()

df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

df


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index,y=df.loc[:,"Close"]))
fig.update_layout(title="Close Price History of AAPL share",xaxis_title="Date",yaxis_title="Close Price USD")
plot(fig)

# considering closing price for analysis

data = df.loc[:,"Close"].values



