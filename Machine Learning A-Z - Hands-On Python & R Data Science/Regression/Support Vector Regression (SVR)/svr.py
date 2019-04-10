# Support Vector Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling - SVR does not have built in scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma='auto')
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results - For SLR
X_grid = np.arange(min(X), max(y), 0.1);
X_grid = X_grid.reshape((len(X_grid), 1));
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red');
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regressor.predict(X_grid)), color = 'blue');
plt.title('Truth or Bluff (SVR)');
plt.xlabel('Position');
plt.ylabel('Salary');
plt.show();