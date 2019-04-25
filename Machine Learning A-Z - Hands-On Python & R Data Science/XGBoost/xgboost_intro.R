# Extreme Gradient Boosting (XGBoost)

# Importing the dataset
dataset = read.csv('Data\\Churn_Modelling.csv')
dataset = dataset[4:ncol(dataset)]

# Encoding categorical variables as factors
dataset$Geography = as.numeric(factor(x = dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(x = dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(0, 1)))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Exited, SplitRatio = 0.8)
training_set = subset(x = dataset, split == TRUE)
test_set = subset(x = dataset, split == FALSE)

# Feature Scaling
training_set[, -11] = scale(training_set[, -11])
test_set[, -11] = scale(test_set[, -11])

# Fitting XGBoost to the Training Set
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]),
                     label = training_set$Exited,
                     nrounds = 10)

# Predicting the Test Set results
y_prob = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = ifelse(y_prob >= 0.5, 1, 0)

# Make the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Applying k-Fold Cross Validation
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(X = folds, FUN = function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]),
                       label = training_set$Exited,
                       nrounds = 10)
  y_prob = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = ifelse(y_prob >= 0.5, 1, 0)
  cm = table(test_fold[, 11], y_pred)
  accuracies = (cm[1, 1] + cm[2, 2]) / (cm[1, 1] + cm[2, 2] + cm[1, 2] + cm[2, 1])
  return(accuracies)
})
mean(as.numeric(cv))
