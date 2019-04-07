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
