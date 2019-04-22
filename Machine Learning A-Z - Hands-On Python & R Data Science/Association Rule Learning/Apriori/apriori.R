# Apriori

# Data Preprocessing
library(arules)
dataset = read.transactions(file = 'Data\\Market_Basket_Optimization.csv',
                            sep = ',',
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 50)

# Training Apriori on the dataset
rules = apriori(data = dataset,
                parameter = list(support = 0.004, #First looking at items bough 4 times a day, 7*3 / 7500
                                 confidence = 0.2))

# Visualize the results
inspect(sort(rules, by = 'lift')[1:10])
