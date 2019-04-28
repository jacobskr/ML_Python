# AutoEncoders

# Importing Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the datasets
movies = pd.read_csv('Data\\ml-1m\\movies.dat', sep='::', header=None,
                     engine='python', encoding='latin-1')
users = pd.read_csv('Data\\ml-1m\\users.dat', sep='::', header=None,
                     engine='python', encoding='latin-1')
ratings = pd.read_csv('Data\\ml-1m\\ratings.dat', sep='::', header=None,
                      engine='python', encoding='latin-1')

# Preparing the training set and the test set using 100k reviews
training_set = pd.read_csv('Data\\ml-100k\\u1.base', delimiter='\t',
                           header=None)
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('Data\\ml-100k\\u1.test', delimiter='\t',
                       header=None)
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = np.unique(np.append(training_set[:,0],test_set[:,0])).size
nb_movies = np.unique(np.append(training_set[:,1],test_set[:,1])).size


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Creating the architecture of the Neural Network
class SAE(nn.Module): # Make a child class of the torch nn.Module
    
    
    def __init__(self, in_features, out_feat1, out_feat2, out_feat3, activation):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(in_features, out_feat1) # First full connection
        self.fc2 = nn.Linear(out_feat1, out_feat2) # Second full connection
        self.fc3 = nn.Linear(out_feat2, out_feat3)  # Third full connection
        self.fc4 = nn.Linear(out_feat3, in_features) # Fourth full connection
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'rectifier':
            self.activation = nn.ReLU()
        else:
           raise Exception("Activation function must be either 'sigmoid' or 'rectifier'")


    def 

    
    
    
    
    
    
# Number of neurons in each hidden layer, can optimize these
in_features = nb_movies
out_feat1 = 20
out_feat2 = 10
out_feat3 = 20
activation = 'sigmoid'
