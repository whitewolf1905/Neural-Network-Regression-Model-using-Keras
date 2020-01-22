import pandas as pd 
import numpy as np 

import keras
from keras.models import Sequential
from keras.layers import Dense


def regression_model():
	model = Sequential()
	model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

concrete_data = pd.read_csv('concrete_data.csv')
# print("sfsdfsd")
concrete_data.shape
concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column
# print(predictors.head())
# print("llb")
target.head()
# print("llasfs")
predictors_norm = (predictors-predictors.mean())/predictors.std()
n_cols = predictors_norm.shape[1] #number of predictors

model = regression_model()

model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)