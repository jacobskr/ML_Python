# Restricted Boltzmann Machine

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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0 # or operator doesn't work with pytorch
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
class RBM():
    
    
    def __init__(self, nv, nh):
        # Weights - number of hidden and visible nodes
        self.W = torch.randn(nh, nv) 
        # Biases
        self.a = torch.randn(1, nh) # 2D tensor coresponding to batch, bias
        self.b = torch.randn(1, nv) # 2D tensor coresponding to batch, bias
    
    
    def sample_h(self, x):
        # Probability of hidden neuron given visible neuron ...  P(h|v)
        wx = torch.mm(x, self.W.t()) # mm is product of two tensors
        activation = wx + self.a.expand_as(wx) # Bias applied to each line of batch
        p_h_given_v = torch.sigmoid(activation) # probability hidden node is activated given the visible node 
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    
    def sample_v(self, y):
        # Probability of hidden neuron given visible neuron ...  P(h|v)
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy) # Bias applied to each line of batch
        p_v_given_h = torch.sigmoid(activation) # probability visible node is activated given the hidden node 
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0) # Trick to keep a tensor of 2 dimensions
        self.a += torch.sum((ph0 - phk), 0)
        
        # Can also add things like learning rate, etc if we want to imporve model.

# Define parameters
nv = len(training_set[0])
nh = 100 # Good starting point for number of features to detect. Can adjust.
batch_size = 100 # Bigger numbers are faster learning, a batch size of 1 is reinforcement learning

# Instantiate RMB
rbm = RBM(nv, nh)

# Train RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0] # Only worried about movies that were rated by user
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('Epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))









