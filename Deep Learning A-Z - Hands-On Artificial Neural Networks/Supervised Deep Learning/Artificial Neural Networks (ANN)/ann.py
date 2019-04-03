# Artificial Neural Network

# Part 1 - Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    # Country
le_X_1 = LabelEncoder()
X[:, 1] = le_X_1.fit_transform(X[:, 1])

    # Gender
le_X_2 = LabelEncoder()
X[:, 2] = le_X_1.fit_transform(X[:, 2])

# Non-ordinal categorical variables (country), so need to create dummy variables
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:] # Remove first dummy variable

# Splitting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)