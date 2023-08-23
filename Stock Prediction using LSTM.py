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
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()

df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

df


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index,y=df.loc[:,"Close"]))
fig.update_layout(title="Close Price History of AAPL share",xaxis_title="Date",yaxis_title="Close Price USD")
plot(fig)

# considering closing price for analysis

data = df.loc[:,"Close"].values

#training Data 95%
training_data_len = int(np.ceil( len(data) * .95 ))

data_reshaped = data.reshape(-1,1)

# scaling Data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_reshaped)


train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# test_data = scaled_data[training_data_len - 60: , :]

