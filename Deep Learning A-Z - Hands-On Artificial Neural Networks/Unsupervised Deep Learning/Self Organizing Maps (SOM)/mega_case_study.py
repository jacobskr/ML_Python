# Mega Case Study - Make a Hybrid Deep Learning Model

# Part 1 - Identify the Frauds with the Self-Organizing Map

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Credit_Card_applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
    # For this example, we need the minisom.py file in our directory
    # Future models we will build from scratch
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0,
              learning_rate=0.5, decay_function=None,
              random_seed=3)

    # Randomly initialize weights
som.random_weights_init(X)

    # Train
som.train_random(data=X, num_iteration=100)

# Visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone();
pcolor(som.distance_map().T);
colorbar();
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2);
show();

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2,8)], mappings[(7,7)]), axis=0)
frauds = sc.inverse_transform(frauds)

# Part 2 Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Make the ANN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
clf = Sequential()

# Adding the input layer and first hidden layer
clf.add(Dense(units=2,
              kernel_initializer='uniform',
              activation='relu',
              input_dim=15))

# Adding the output layer
    # If multiclass problem, need softmax instead of sigmoid
clf.add(Dense(units=1,
              kernel_initializer='uniform',
              activation='sigmoid'))

# Compiling the ANN (apply stochastic gradient descent)
clf.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

# Fitting the ANN to the training set
clf.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the probabilities of fraud
y_pred = clf.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
