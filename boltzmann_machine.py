"""

Original file is located at
    https://colab.research.google.com/drive/1JxIEe_TAcz-utfLKTqjbo3foOtqDGD0s

#Boltzmann Machine

#Importing the libraries
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data

# Importing the dataset

# We won't be using this dataset.
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies

user_num = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
movie_num = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns

def convert(data):
  new_data = []
  for user_id in range(1, user_num + 1):
    movie_id = data[:, 1] [data[:, 0] == user_id]
    rating_id = data[:, 2] [data[:, 0] == user_id]
    ratings = np.zeros(movie_num)
    ratings[movie_id - 1] = rating_id
    new_data.append(list(ratings))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network

class RBM():
  def __init__(self, visable_node_num, hidden_node_num):
    self.W = torch.randn(hidden_node_num, visable_node_num)
    # first bias
    self.a = torch.randn(1, hidden_node_num)
    # second bias
    self.b = torch.randn(1, visable_node_num)
  def hidden_sample(self, x):
    # weight times the x neutron
    w_x_neutron = torch.mm(x, self.W.t())
    activation = w_x_neutron + self.a.expand_as(w_x_neutron)
    # probabliity of hidden node is activated given the visable node
    p_h_given_v = torch.sigmoid(activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)
  def visual_sample(self, y):
    wy = torch.mm(y, self.W)
    activation = wy + self.b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h)
  def train(self, v0, vk, ph0, phk):
    self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    self.b += torch.sum((v0 - vk), 0)
    self.a += torch.sum((ph0 - phk), 0)
visable_node_num = len(training_set[0])
hidden_node_num = 100
batch_size = 100
rbm = RBM(visable_node_num, hidden_node_num)

# Training the RBM

epoch_num = 10
for epoch in range(1, epoch_num + 1):
  train_loss = 0
  s = 0.
  for id_user in range(0, user_num - batch_size, batch_size):
    vk = training_set[id_user : id_user + batch_size]
    v0 = training_set[id_user : id_user + batch_size]
    ph0,_ = rbm.hidden_sample(v0)
    for k in range(10):
      _,hk = rbm.hidden_sample(vk)
      _,vk = rbm.visual_sample(hk)
      vk[v0<0] = v0[v0<0]
    phk,_ = rbm.hidden_sample(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
    s += 1.
  print(' loss: '+str(train_loss/s))

# Testing the RBM

test_loss = 0
s = 0.
for id_user in range(user_num):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.hidden_sample(v)
        _,v = rbm.visual_sample(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))