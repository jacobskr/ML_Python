# Principal Component Analysis (PCA)

# Importing the dataset
dataset = read.csv('Data\\Social_Network_Ads.csv')
dataset = dataset[3:5]

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Purchased, SplitRatio = 0.8)
training_set = subset(x = dataset, split == TRUE)
test_set = subset(x = dataset, split == FALSE)

# Feature Scaling
training_set[, 1:(ncol(training_set) - 1)] = scale(training_set[, 1:(ncol(training_set) - 1)])
test_set[, 1:(ncol(test_set) - 1)] = scale(test_set[, 1:(ncol(test_set) - 1)])

# Applying Kernel PCA
library(kernlab)
kpca = kpca(x = ~ .,
            data = training_set[-3],
            kernel = 'rbfdot',
            features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set[-3]))
test_set_pca$Purchased = test_set$Purchased

# Fitting SVM to the Training set
classifier = glm(formula = Purchased ~ .,
                 data = training_set,
                 family = binomial)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualize the Training set results
library(ElemStatLearn)
set =  training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_grid = ifelse(prob_pred > 0.5, 1, 0)
plot(set[, -3],
     main = 'SVM (Training Set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3',
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualize the Test set results
set =  test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_grid = ifelse(prob_pred > 0.5, 1, 0)
plot(set[, -3],
     main = 'SVM (Test Set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3',
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))
