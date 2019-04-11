# Decision Tree Regression

# Regression Template

# Importing the dataset
dataset = read.csv('Data\\Position_Salaries.csv')
dataset = dataset[2:3]

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Decision Tree Regression Model to the dataset
library(rpart)
regressor = rpart(Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))


# Visualising the Decision Tree Regression Model results (for higher resolution and smoother curve)
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