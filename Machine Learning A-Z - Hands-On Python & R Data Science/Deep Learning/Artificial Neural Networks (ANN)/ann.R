# Artificial Neural Network (ANN)

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

# Fitting ANN to the Training Set
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

# Predicting the Test Set results
prod_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = as.vector(prod_pred > 0.5)

# Make the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Disconnect
h2o.shutdown()

