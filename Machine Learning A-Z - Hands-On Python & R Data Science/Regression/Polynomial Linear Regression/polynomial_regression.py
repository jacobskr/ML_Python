# Polynomial Linear Regression

# Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # Trick to get X as matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
    # Only 10 observations, doesn't make sense to do this.

# Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X) # This automatically created column of ones (intercept)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing Simple Linear Regression to the dataset
plt.scatter(X, y, color='red');
plt.plot(X, lin_reg.predict(X), color='blue');
plt.title('Truth or Bluff (Simple Linear Regression)');
plt.xlabel('Position Level');
plt.ylabel('Salary');
plt.show();

# Visualizing Polynomial Linear Regression to the dataset
plt.scatter(X, y, color='red');
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue');
plt.title('Truth or Bluff (Polynomial Linear Regression)');
plt.xlabel('Position Level');
plt.ylabel('Salary');
plt.show();

# Try Degree 3
# Fitting Polynomial Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_reg3.fit_transform(X) # This automatically created column of ones (intercept)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly_3, y)

# Visualizing Polynomial Linear Regression to the dataset
plt.scatter(X, y, color='red');
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color='blue');
plt.title('Truth or Bluff (Polynomial Degree 3 Linear Regression)');
plt.xlabel('Position Level');
plt.ylabel('Salary');
plt.show();

# Using X_grid to get continuous curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red');
plt.plot(X_grid, lin_reg3.predict(poly_reg3.fit_transform(X_grid)), color='blue');
plt.title('Truth or Bluff (Polynomial Degree 3 Linear Regression)');
plt.xlabel('Position Level');
plt.ylabel('Salary');
plt.show();

# Predicting a new result with Simple Linear Regression
print('Estimated salary using SLR = $',
      round(lin_reg.predict([[6.5]])[0],2))

# Predicting a new result with degree 2 Polynomial Linear Regression
print('Estimated salary using PLR = $',
      round(lin_reg2.predict(poly_reg.transform([[6.5]]))[0],2))

# Predicting a new result with degree 3 Polynomial Linear Regression
print('Estimated salary using PLR = $',
      round(lin_reg3.predict(poly_reg3.transform([[6.5]]))[0],2))
