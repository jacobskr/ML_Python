# Reinforcement Learning - Upper Confidence Bound (UCB)

# Importing the dataset
dataset = read.csv('Data\\Ads_CTR_Optimization.csv') # Simulation dataset in place of actual experiment

# Implimenting UCB from scratch
N = 10000
d = 10
ads_selected = integer()

  # Step 1
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0

  # Step 2
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    if (numbers_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
      upper_bound = 1e400
    }
  # Step 3
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  } 
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualizing the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of Ad Selections',
     xlab = 'Ad',
     ylab = 'Number of Times Each Ad Was Selected')
