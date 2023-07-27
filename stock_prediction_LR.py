# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:15:35 2023

@author: nehak
"""

import numpy as np 
import pandas as pd


# import all stock prices 
data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\prices.csv", index_col = 0)

data.head(5)

data = data[data.symbol == 'GOOG']

data.describe()


data.isnull().sum()



