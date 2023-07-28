# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:12:12 2023

@author: nehak
"""

import numpy as np
import pandas as pd

df_technical = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\prices-split-adjusted.csv")
df_fundamental = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\fundamentals.csv")
df_fundamental = df_fundamental.drop("Unnamed: 0", axis=1)
df_securities = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\securities.csv")


print(df_technical)
print(df_technical.info())
df_technical.describe()

print(df_fundamental)
print(df_fundamental.info())
df_fundamental.describe()

print(df_securities)
print(df_securities.info())
df_securities.describe()