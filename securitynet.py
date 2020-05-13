import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

#Define parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
batch_size = 32
epochs = 50
split = 0.8



#TODO import and clean data
data = pd.read_csv('spy.us.txt')

high = data['High']
high = high[1:]

del data['High']
del data['OpenInt']
del data['Date']

data = data.drop(index = 3200)

join = zip(data,high)
join = np.array(join)

joined = pd.DataFrame(data=data, columns = ['in','high'])

joined = joined.sample(frac=1)

train = joined.sample(frac=split, axis=0)
train_y = train['high']
del train['high']
train_x = train.values

val = joined.drop(train.index)
val_y = val['high']
del val['high']
val_x = val.values

# print(train_x.shape)
# train_x = train_x.reshape(-1, 4)
# train_x = train_x.astype('float')

# val_x = np.concatenate(val_x)
# val_x = val_x.reshape(-1, 4)
# val_x = val_x.astype('float')

train_y = train_y.values
val_y = val_y.values

train_x = torch.from_numpy(np.float32(train_x))
train_y = torch.from_numpy(np.float32(train_y))
val_x = torch.from_numpy(np.float32(val_x))
val_y = torch.from_numpy(np.float32(val_y))

train_data = torch.utils.data.TensorDataset(train_x, train_y)
train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data = torch.utils.data.TensorDataset(val_x, val_y)
val_load = torch.utils.data.DataLoader(val_data, batch_size=batch_size)


#TODO build net
class NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.d = None
		self.fc1 = nn.Linear(4, 16)
		self.fc2 = nn.Linear(16,24)
		self.fc3 = nn.Linear(24,32)
		self.fc4 = nn.Linear(32,18)
		self.fc5 = nn.Linear(18,1)


	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		return x

# def learn(loss):
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()

#  def accuracy(out, pred):
#  	#TODO calc accuracy
#  	return accuracy

#  def train(model, input, expected_output):
#  	input = input.to(device)
#  	output = output.to(device)

#  	#TODO do training

#  def validate(model, input, expected_output):
#  	input = input.to(device)
#  	output = output.to(device)

#  	#TODO do validation

#   model = NN().to(device)
#   optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
#   criterion=nn.mse()

#   train(model, train_x, train_y)
#   validate(model, val_x, val_y)

nn = NN()
