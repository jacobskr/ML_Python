# Support Vector Regression

# Importing the dataset
dataset = read.csv('Data\\Position_Salaries.csv')
dataset = dataset[2:3]

# Feature Scaling - e1071 automatically does scaling

# Fitting the Support Vector Regression Model to the dataset
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')

# Predicting a new result
y_pred = predict(regressor, data.frame( Level = 6.5))

# Visualising the Support Vector Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Support Vector Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff') +
  xlab('Level') +
  ylab('Salary')
