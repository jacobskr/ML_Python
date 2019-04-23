# Reinforcement Learning - Upper Confidence Bound (UCB)

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Ads_CTR_Optimization.csv') # Simulation dataset in place of actual experiment

# Implementing UCB from scratch
import math
N = 10000 # Number of iterations
d = 10 # Number of ads
ads_selected = []

    # Step 1
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

    # Step 2
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0): # Make sure we go through each of the ads first
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt((3 / 2) * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else: 
            upper_bound = 1e400
    # Step 3
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualize the results
plt.hist(ads_selected);
plt.title('Histogram of Ad Selections');
plt.xlabel('Ads');
plt.ylabel('Number of Times Selected');
plt.show();