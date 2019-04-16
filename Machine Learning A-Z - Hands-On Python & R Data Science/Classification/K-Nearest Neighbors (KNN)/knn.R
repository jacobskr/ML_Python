# K-Nearest Neighbors

# Importing the dataset
dataset = read.csv('Data\\Social_Network_Ads.csv')
dataset = dataset[3:5]

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Purchased, SplitRatio = 0.75)
training_set = subset(x = dataset, split == TRUE)
test_set = subset(x = dataset, split == FALSE)

# Feature Scaling
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

# Fitting KNN to the Training set and predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -3],
             test = test_set[, -3],
             cl = training_set[, 3],
             k = 5)


# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualize the Training set results
library(ElemStatLearn)
set =  training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[, -3],
               test = grid_set,
               cl = training_set[, 3],
               k = 5)
plot(set[, -3],
     main = 'KNN (Training Set)',
     xlab = 'Age', ylab = 'Esimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3], 'green4', 'red3'))

# Visualize the Test set results
set =  test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[, -3],
             test = grid_set,
             cl = training_set[, 3],
             k = 5)
plot(set[, -3],
     main = 'KNN (Test Set)',
     xlab = 'Age', ylab = 'Esimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3], 'green4', 'red3'))
