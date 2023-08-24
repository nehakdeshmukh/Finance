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
        
        
x_train=np.array(x_train)
y_train=np.array(y_train)


# reshaped data for LSTM

X_train_reshaped = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# test_data = scaled_data[training_data_len - 60: , :]

from keras.models import Sequential
from keras.layers import LSTM,Dense

model= Sequential()
model.add(LSTM(256,return_sequences=True,input_shape=(X_train_reshaped.shape[1],1)))
# model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))


# compile model

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train_reshaped,y_train,batch_size=10,epochs=5)

# Create Test Data 
test_data = scaled_data[training_data_len - 60: , :]

x_test=[]
y_test = data[training_data_len::]


# Create test data 
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    

x_test=np.array(x_test)

# reshape array 
x_test_reshape = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# prediction 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# model.evaluate(x_test,predictions)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predictions))


# plot for prediction  
train = df[:training_data_len]
validation = df[training_data_len:]
validation['prediction']=predictions

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index,y=df.loc[:,"Close"],name="train"))
fig.add_trace(go.Scatter(x=validation.index,y=validation.loc[:,"Close"],name="validation"))
fig.add_trace(go.Scatter(x=validation.index,y=validation["prediction"],name="predictions"))
fig.update_layout(title="Close Price History of AAPL share",xaxis_title="Date",yaxis_title="Close Price USD")
plot(fig)