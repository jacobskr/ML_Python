# Simple Linear Regression

# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data\\Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Salary, SplitRatio = 2/3)
training_set = subset(x = dataset, split == TRUE)
test_set = subset(x = dataset, split == FALSE)

# Feature Scaling - Don't need to do, library takes care of it for us

# Run Regression

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Predict the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualize the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = YearsExperience, y = Salary),
             data = training_set,
             color = 'red') +
  geom_line(aes(YearsExperience, y = predict(regressor, newdata = training_set)),
            data = training_set,
            color = 'blue') +
  geom_smooth(data= training_set, aes(x=YearsExperience, y= Salary), method = "lm")+
  ggtitle('Salary vs. Experience (Training Set)') +
  xlab('Years Experience') +
  ylab('Salary')

# Visualize the Test set results
ggplot() +
  geom_point(aes(x = YearsExperience, y = Salary),
             data = test_set,
             color = 'red') +
  geom_line(aes(YearsExperience, y = predict(regressor, newdata = training_set)),
            data = training_set,
            color = 'blue') +
  geom_smooth(data= training_set, aes(x=YearsExperience, y= Salary), method = "lm")+
  ggtitle('Salary vs. Experience (Training Set)') +
  xlab('Years Experience') +
  ylab('Salary')
  