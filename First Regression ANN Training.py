# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 17:47:35 2021

@author: msk2766
"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv(
    r'C:\Users\msk2766\Documents\Mohamed Khaled\Documents\Coding Training\Python Training\heart.data.csv')
print(df.head())

df.isna().sum()

df = df.drop("Unnamed: 0", axis=1)
# A few plots in Seaborn to understand the data

plt.figure()
sns.lmplot(x='biking', y='heart.disease', data=df)
plt.figure()
sns.lmplot(x='smoking', y='heart.disease', data=df)
plt.figure()
sns.pairplot(df[["heart.disease", "biking", "smoking"]], diag_kind="kde")


train_stats = df.describe()
train_stats.pop("heart.disease")
train_stats = train_stats.transpose()
train_stats


x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

x = x_df.to_numpy()
y = y_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# Build the network
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())


history = model.fit(X_train, y_train, verbose=1, epochs=500,
                    validation_data=(X_test, y_test))


prediction_test = model.predict(X_test)
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =",
      np.mean(prediction_test-y_test)**2)


prediction_train = model.predict(X_train)
print(y_train, prediction_train)
print("Mean sq. errror between y_train and predicted =",
      np.mean(prediction_train-y_train)**2)

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.figure
plt.scatter(y_train, prediction_train, c='crimson')
# plt.yscale('log')
# plt.xscale('log')
p1 = max(max(y_train), max(prediction_train))
p2 = min(min(y_train), min(prediction_train))
plt.plot([p1, p2], [p1, p2], 'y-')
plt.xlabel('Actual', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


plt.figure
plt.scatter(y_test, prediction_test, c='crimson')
# plt.yscale('log')
# plt.xscale('log')
p1 = max(max(y_test), max(prediction_test))
p2 = min(min(y_test), min(prediction_test))
plt.plot([p1, p2], [p1, p2], 'y-')
plt.xlabel('Actual', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
