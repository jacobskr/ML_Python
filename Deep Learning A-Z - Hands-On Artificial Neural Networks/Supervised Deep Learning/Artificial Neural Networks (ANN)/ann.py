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

# Part 4 - Evaluating, improving, and tuning ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    clf = Sequential()
    clf.add(Dense(units=6, kernel_initializer='uniform',activation='relu', input_dim=11))
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return clf


clf = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=clf,
                             X=X_train, y=y_train,
                             cv=5, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

# Dropout regularization to reduce overfitting if needed
    #If you have overfitting, suggest to add dropout to all hidden layers
    #Start with p=0.1 and if still overfitting, go up
# from keras.layers import Dropout
# add 'clf.add(Dropout(p=0.1))' after every hidden layer add


# Tuning the ANN
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    clf = Sequential()
    clf.add(Dense(units=6, kernel_initializer='uniform',activation='relu', input_dim=11))
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return clf


clf = KerasClassifier(build_fn=build_classifier)

params = {'batch_size': [25, 32],
          'epochs': [100, 500],
          'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=clf,
                           param_grid=params,
                           scoring='accuracy',
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_




