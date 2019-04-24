# Artificial Neural Network (ANN)

# Part 1 - Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X_1 = LabelEncoder()
X[:, 1] = le_X_1.fit_transform(X[:, 1])
le_X_2 = LabelEncoder()
X[:, 2] = le_X_2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Part 2 - Make the ANN

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
    # Starting with (# nodes input layer + # nodes output layer) / 2
        # (11 + 1) / 2 = 6 for our data
    # Recommended to use rectified linear unit (relu) for hidden layers and sigmoid for output
classifier.add(Dense(units=6, # Output
                     kernel_initializer='uniform',
                     activation='relu',
                     input_dim=11)) # Input

# Adding the second hidden layer
    # Don't need to specify input_dim because only need to spevidy in first layer
classifier.add(Dense(units=6, # Output
                     kernel_initializer='uniform',
                     activation='relu'))

# Adding the output layer
    # If multiclass problem, need softmax instead of sigmoid
classifier.add(Dense(units=1, # Output
                     kernel_initializer='uniform',
                     activation='sigmoid'))

# Compiling the ANN (apply stochastic gradient descent)
    # adam is a method of stochastic gradient descent
    # binary/categorical_crossenntropy is same as the "SSE" function for logit
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Make predictions and evaluate the model

# Predicting the Test Set results
y_pred = classifier.predict(X_test)

# Update to binary using threshhold of 0.5
y_pred = (y_pred > 0.5)

# Making the confusion matrix and finding accuracy
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
accuracy_score(y_test, y_pred)