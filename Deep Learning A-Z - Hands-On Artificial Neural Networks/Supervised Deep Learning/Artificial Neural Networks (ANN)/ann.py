# Artificial Neural Network

# Part 1 - Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Part 2 - Make the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
clf = Sequential()

# Adding the input layer and first hidden layer
    # Starting with (# nodes input layer + # nodes output layer) / 2
        # (11 + 1) / 2 = 6 for our data
    # Recommended to use rectified linear unit (relu) for hidden layers and sigmoid for output
clf.add(Dense(units=6,
              kernel_initializer='uniform',
              activation='relu',
              input_dim=11))

# Add the second hidden layer
    # Don't need to specify input_dim since it already knows what to expect
clf.add(Dense(units=6,
              kernel_initializer='uniform',
              activation='relu'))

# Adding the output layer
    # If multiclass problem, need softmax instead of sigmoid
clf.add(Dense(units=1,
              kernel_initializer='uniform',
              activation='sigmoid'))

# Compiling the ANN (apply stochastic gradient descent)
    # adam is a method of stochastic gradient descent
    # binary/categorical_crossenntropy is same as the "SSE" function for logit
clf.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

# Fitting the ANN to the training set
clf.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
    # Gives probabilities
y_pred = clf.predict(X_test)

# Update to binary using threshhold of 0.5
y_pred = (y_pred > 0.5)

# Making the confusion matrix and finding accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
accuracy_score(y_test, y_pred)

# Next step could be to do parameter training to get higher accuracy

# =============================================================================
# Homework: 
# Use our ANN model to predict if the customer with the following informations will leave the bank: 
# 
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000
# So should we say goodbye to that customer ?
# =============================================================================

hmwk = clf.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3,
                                           60000, 2, 1, 1, 50000]])))
hmwk > 0.5


