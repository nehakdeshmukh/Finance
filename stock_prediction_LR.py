# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:15:35 2023

@author: nehak
"""

import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# import all stock prices 
data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\prices.csv", index_col = 0)

data.head(5)

data = data[data.symbol == 'GOOG']

data.describe()


data.isnull().sum()


predict_col = 'close'
predict_out = 5
test_size = 0.2



label = data[predict_col].shift(-predict_out)
X = np.array(data[[predict_col]])
X = preprocessing.scale(X)
X_lately = X[-predict_out:]
X = X[:-predict_out]
label.dropna(inplace=True)
y = np.array(label)



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size,random_state=3)

lr = LinearRegression()
lr.fit(X_train,y_train)

score = lr.score(X_test,y_test)
score

predict = lr.predict(X_lately)
predict
