# Eclat

# Data Preprocessing
library(arules)
dataset = read.transactions(file = 'Data\\Market_Basket_Optimization.csv',
                            sep = ',',
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
rules = eclat(data = dataset,
                parameter = list(support = 0.004, #First looking at items bough 4 times a day, 7*3 / 7500
                                 minlen = 2))

# Visualize the results
inspect(sort(rules, by = 'support')[1:10])
