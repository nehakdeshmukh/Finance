# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:06:16 2023

@author: nehak
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.preprocessing
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf


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


# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

# function to create train, validation, test data given stock data and sequence length
def split_data(stock, seq_len,valid_set_size_percentage,test_set_size_percentage):
    #data_raw = stock.as_matrix() # convert to numpy array
    data_raw = np.array(df_stock_norm)
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


# create train, test data
seq_len = 20 # choose sequence length


x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(df_stock_norm, seq_len,valid_set_size_percentage,test_set_size_percentage)

print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)


# AAPL stock 
fig = go.Figure()
fig.add_trace(go.Scatter(y= df_stock_norm.open.values,name='open'),)
fig.add_trace(go.Scatter(y= df_stock_norm.close.values,name='close'))
fig.add_trace(go.Scatter(y= df_stock_norm.low.values,name='low'))
fig.add_trace(go.Scatter(y= df_stock_norm.high.values,name='high'))
fig.update_layout(title="Stock Price",xaxis_title="time in days",  yaxis_title="Price")
fig.show()


## Basic Cell RNN in tensorflow

index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]



# parameters
n_steps = seq_len-1 
n_inputs = 4 
n_neurons = 200 
n_outputs = 4
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100 
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]




# tf.reset_default_graph()
ops.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# tf.compat.v1.nn.rnn_cell.BasicRNNCell(
# use Basic RNN Cell
layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
          for layer in range(n_layers)]

# tf.compat.v1.nn.rnn_cell.MultiRNNCell(
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
loss = tf.reduce_mean(tf.square(outputs - y)) # loss function = mean squared error 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)

