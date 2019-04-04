# Recurrent Neural Network

# Part 1 - Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the training set
dataset_train = pd.read_csv('Data\\Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values #Trick to get np array of 1 column, 1:2

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)    

# Reshaping - This will let us add more indicators later if wanted
    # Shape will be (# of timestamps, # of features, # of indicators)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential()

# Adding the first LSTM layer and some dropout regularization to reduce overfitting
    # Need return_sequences=True when adding more after
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # Suggested dropout rate of 20%
regressor.add(Dropout(rate=0.2))

# Adding the second LSTM layer and some dropout regularization to reduce overfitting
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the third LSTM layer and some dropout regularization to reduce overfitting
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the fourth LSTM layer and some dropout regularization to reduce overfitting
    # Need return_sequences=False when not adding any more
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compile the RNN
    # RMSprop is usually good for RNN optimizer, adam also good choice
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(x=X_train, y=y_train, epochs=100, batch_size=32)

# Part 3 - Make predictions and visualize results

