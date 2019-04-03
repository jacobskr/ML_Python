# Convolutional Neural Network

# Don't need to do any preprocessing because of how the data is set up in folders.

# Part 1 - Building the CNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
clf = Sequential()

    # Step 1 - Convolution
        # (64, 64, 3) for Theano backend, (3, 64, 64) for Tensorflow backend
clf.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Flattening
clf.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Step 3 - Flattening
clf.add(Flatten())

    # Step 4 - Full Connection
        #Our rule for output_dim from ANN, is annoying to calculate for this, so we chose 128 for now.
clf.add(Dense(activation='relu', units=128))
