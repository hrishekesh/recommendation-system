#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:54:15 2018

@author: hrishekesh.shinde
"""

from flask import Flask
from flask import request
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

app = Flask(__name__)
CORS(app)

#Read ratings data
indexed_ratings = pd.read_csv('indexed_ratings.csv')
indexed_ratings['userId'] = indexed_ratings['userId'].astype(int)
indexed_ratings['movieId'] = indexed_ratings['movieId'].astype(int)
indexed_ratings['movieIndex'] = indexed_ratings['movieIndex'].astype(int)

max_movie_index = max(indexed_ratings['movieIndex'])
print(max_movie_index)

movies = pd.read_csv('selected_movies.csv')

def loadmodel():
    model = SAE()
    model.load_state_dict(torch.load('modelCheckpoints/nnmodel.ae'))
    print('model loaded')
    #print(model.state_dict())
    model.eval()
    return model

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
test_loss = 0
s = 0.
sae = loadmodel()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

def formatData():
    return movies['userrating']

def predictRatings():
    rowData = formatData()
    rowTorch = torch.FloatTensor(rowData)
    print(rowTorch)
    inputData = Variable(rowTorch).unsqueeze(0)
    target_ratings = inputData[:, :movies.shape[0]]
    output = sae(inputData)
    output_ratings = output[:, :movies.shape[0]]
    
    inputData.require_grad = False
    loss = criterion(output_ratings, target_ratings)
    mean_corrector = max_movie_index/float(torch.sum(inputData.data) + 1e-10)
    single_loss = 0
    single_loss += np.sqrt(loss.data[0]*mean_corrector)
    print(single_loss)
    output_rating_rounded = np.around(output_ratings.detach().numpy()[0], decimals=1 )
    output_rating = np.round(output_ratings.detach().numpy()[0] * 2) / 2
    movies['predictedrating'] = output_rating[:movies.shape[0]]

    

def convertJsonToArray(dataDict):
    movies['userrating'] = np.zeros(movies.shape[0])
    for rating in dataDict['ratings']:
        movies['userrating'][movies['movieId'] == rating['movieId']] = rating['rating']
        
'''
request object format and unit testing code

testDict = {"ratings": [{"movieId": 117466,"rating": 5.0},
                        {"movieId": 122892,"rating": 4.0},
                        {"movieId": 128838,"rating": 3.0},
                        {"movieId": 129937,"rating": 2.0},
                        {"movieId": 133365,"rating": 1.0},
                        {"movieId": 135532,"rating": 0.5},
                        {"movieId": 137383,"rating": 1.5},
                        {"movieId": 174745,"rating": 2.5},
                        {"movieId": 176563,"rating": 3.5},
                        {"movieId": 176935,"rating": 4.5}]}

convertJsonToArray(testDict)
predictRatings()'''
        
@app.route('/api/v1/recommendations/movies', methods=['POST'])
def getMovieRecommendations():
    if not request.json:
        return 'No request data', 400
    user_rating_data_dict = request.json
    convertJsonToArray(user_rating_data_dict)
    predictRatings()
    return movies.to_json(orient='records'), 200

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8086, app)
