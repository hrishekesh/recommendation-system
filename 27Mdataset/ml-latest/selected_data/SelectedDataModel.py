#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:43:58 2018

@author: hrishekesh.shinde
"""

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Read ratings data
indexed_ratings = pd.read_csv('indexed_ratings.csv')
indexed_ratings['userId'] = indexed_ratings['userId'].astype(int)
indexed_ratings['movieId'] = indexed_ratings['movieId'].astype(int)
indexed_ratings['movieIndex'] = indexed_ratings['movieIndex'].astype(int)

max_movie_index = max(indexed_ratings['movieIndex'])
print(max_movie_index)

movies = pd.read_csv('selected_movies.csv')





class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(max_movie_index+1, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, max_movie_index+1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
# create a neural network instance
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

def savemodel(model):
    print('saving model')
    #print(model.state_dict())
    torch.save(model.state_dict(), 'modelCheckpoints/nnmodel.ae')
    
def loadmodel():
    model = SAE()
    model.load_state_dict(torch.load('modelCheckpoints/nnmodel.ae'))
    print('model loaded')
    #print(model.state_dict())
    model.eval()
    return model

    
number_of_users = indexed_ratings['userId'].unique()    
    
rating_matrix = []
for user in indexed_ratings['userId'].unique():
    # user rating columns
    user_rating_data = np.zeros(max_movie_index+1)
    user_rating_data[indexed_ratings['movieIndex'][indexed_ratings['userId'] == user]] = indexed_ratings['rating'][indexed_ratings['userId'] == user]
       
    rating_matrix.append(list(user_rating_data))
        
# split data in training and testing sets

num_users = len(rating_matrix)
print('num_users')
print(num_users)
training_set_num = int(0.8 * num_users)
print('training_set_num')
print(training_set_num)
testing_set_num = int(num_users) - training_set_num
print('testing_set_num')
print(testing_set_num)
    
# split data in training and testing sets
training_set = rating_matrix[:training_set_num]
testing_set = rating_matrix[training_set_num:]
print(len(training_set))
print(len(testing_set))
    
# create torch tensors
training_set_torch = torch.FloatTensor(training_set)
test_set_torch  = torch.FloatTensor(testing_set)
    
# train the neural network
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(len(training_set)):
        input = Variable(training_set_torch[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = (max_movie_index+1)/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
savemodel(sae) 
        
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(len(testing_set)):
    input = Variable(test_set_torch[id_user]).unsqueeze(0)
    target = Variable(test_set_torch[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = (max_movie_index+1)/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))