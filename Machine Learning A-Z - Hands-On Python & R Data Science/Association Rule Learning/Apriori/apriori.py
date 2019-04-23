# Apriori

# Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Market_Basket_Optimization.csv', header=None)

# Need to turn data into a list of lists
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0, len(dataset.columns))])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions,
                min_support=0.003, # First looking at items bough 3 times a day, (7*3)/7500
                min_confidence=0.2,
                min_lift=3,
                min_length=2)

# Visualizing the results
results = list(rules)
clean_results = []
for i in range(0, len(results)):
    result_dict = dict()
    result_dict["RULE"] = list(results[i][0])
    result_dict["SUPPORT"] = results[i][1]
    result_dict["CONFIDENCE"] = results[i][2][0][2]
    result_dict["LIFT"] = results[i][2][0][3]
    clean_results.append(result_dict)