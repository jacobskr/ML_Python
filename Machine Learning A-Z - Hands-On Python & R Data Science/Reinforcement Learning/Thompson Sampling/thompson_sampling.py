# Reinforcement Learning - Thompson Sampling

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data\\Ads_CTR_Optimization.csv') # Simulation dataset in place of actual experiment

# Implementing Thompson Sampling from scratch
import random
N = 10000 # Number of iterations
d = 10 # Number of ads
ads_selected = []

    # Step 1
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

    # Step 2
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)
    # Step 3
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
# Visualize the results
plt.hist(ads_selected);
plt.title('Histogram of Ad Selections');
plt.xlabel('Ads');
plt.ylabel('Number of Times Selected');
plt.show();