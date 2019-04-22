# K-Means Clustering

# Import dataset
dataset <- read.csv('Data\\Mall_Customers.csv')
X <- dataset[4:5]

# Use elbow method to find optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('Clusters of Clients'), xlab = 'Number of Clusters', ylab = 'WCSS')

# Applying k-means to the dataset using number of clusters found above
set.seed(29)
kmeans = kmeans(X, centers = 5, iter.max = 300, nstart = 10)

# Visualizing the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of Clients'),
         xlab = 'Annual Income',
         ylab = 'Soending Score')
