# Random Forest Regression

# Importing the dataset
dataset = read.csv('Data\\Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting the Random Forest Regression Model to the dataset
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], # Need to give x as a dataframe (that is what [] does)
                         y = dataset$Salary, # Need to give y as a vector (that is what $ does)
                         ntree =500)

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Random Forest Regression Model results
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff') +
  xlab('Level') +
  ylab('Salary')
