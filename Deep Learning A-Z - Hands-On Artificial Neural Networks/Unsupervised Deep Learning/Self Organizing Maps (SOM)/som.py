# Self Organizing Maps

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
              learning_rate=0.5, decay_function=None)

    # Randomly initialize weights
som.random_weights_init(X)

    # Train
som.train_random(data=X, num_iteration=100)

# Visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
