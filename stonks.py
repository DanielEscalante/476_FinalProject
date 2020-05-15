#using LSTM to predict future stonk price

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

df = pd.read_csv("DIA.csv")

train, test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)

#scale and build array
train_cols = ["Open", "High", "Close", "Volume"]
x = train.loc[:,train_cols].values
scale = MinMaxScaler()
x_train = scale.fit_transform(x)
x_test = scale.transform(test.loc[:,train_cols])

#hyper-params
TIME_STEPS = 10 #this is how many days of data we use to predict the next day
BATCH_SIZE = 24
LEARNING_RATE = 0.0002
EPOCHS = 30

#build timeseries(use past days to predict next day), y_col is index of output column
def build_ts(mat, y_col):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col]
    return x, y

#data has to be divisble by batch size
def trim(mat, batch_size):
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

#create training and validation data
x_t, y_t = build_ts(x_train, 1)
x_t = trim(x_t, BATCH_SIZE)
y_t = trim(y_t, BATCH_SIZE)
x_t2, y_t2 = build_ts(x_test, 1)

#the model
lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), stateful=True))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(50,activation='relu'))
lstm_model.add(Dense(40,activation='relu'))
lstm_model.add(Dense(30,activation='relu'))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='linear'))
optimizer = keras.optimizers.RMSprop(lr=LEARNING_RATE)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

history = lstm_model.fit(x_t, y_t, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)
print(history)

BATCH_SIZE=24
x_val, x_test_t = np.split(trim(x_t2, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim(y_t2, BATCH_SIZE),2)
x_val = trim(x_val, BATCH_SIZE)
y_val = trim(y_val, BATCH_SIZE)
eval = lstm_model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
print(eval)

