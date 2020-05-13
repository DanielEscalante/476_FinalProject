import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

#Define parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#TODO import and clean data

#TODO build net
class NN(nn.Module):
	def __init__(self):
		super.__init__()
		self.net=None

	def forward(self, x):
		#TODO do network stuff
		return x

def learn(loss):
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

 def accuracy(out, pred):
 	#TODO calc accuracy
 	return accuracy

 def train(model, input, expected_output):
 	input = input.to(device)
 	output = output.to(device)

 	#TODO do training

 def validate(model, input, expected_output):
 	input = input.to(device)
 	output = output.to(device)

 	#TODO do validation

  model = NN().to(device)
  optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion=nn.mse()

  train(model, train_x, train_y)
  validate(model, val_x, val_y)


