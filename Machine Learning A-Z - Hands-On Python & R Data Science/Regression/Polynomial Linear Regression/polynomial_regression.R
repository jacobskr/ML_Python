# Polynomial Linear Regression

# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data\\Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
  # Only 10 observations, doesn't make sense to do this.

# Fitting Simple Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ Level,
             data = dataset)


# Fitting Polynomial Linear Regression to the dataset
dataset$Level2 = dataset$Level ** 2
dataset$Level3 = dataset$Level ** 3
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualizing the Simple Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth or Bluff (Simple Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualizing the degree 3 Polynomial Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Simple Linear Regression
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Predicting a new result with Polynomial Linear Regression
y_pred2 = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                 Level2 = 6.5 ** 2,
                                                 Level3 = 6.5 **3))
