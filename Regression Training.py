# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 15:23:28 2021

@author: msk2766
"""

import pandas 
import keras
from keras.models import sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

...
# load dataset
dataframe = pandas.read_csv(r'C:\Users\msk2766\Documents\Mohamed Khaled\Documents\Coding Training\Python Training\housing.csv', delim_whitespace=True, header=None)
dataset = dataframe.values
print(dataset.head())

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]



