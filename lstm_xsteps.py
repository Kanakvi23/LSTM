import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras import optimizers
import h5py

# fix random seed reproducibility
np.random.seed(88)

# Locate csv file to be read and import
fname = "H:\\data\\ois.csv"
data_csv = pd.read_csv(fname)

# Moving averae for m days
m = 20
MA_m = data_csv.rolling(window = m).mean()
data_csv = MA_m

# Drop rows with Nan
data_csv = data_csv.dropna()

# Remove date column
data_csv = data_csv.drop('date', axis = 1)
dataset = data_csv.values
dataset = dataset.astype('float32')

# Train and test sets division
data_len = len(dataset)
train_frac = 0.8
train_end = math.ceil(train_frac*data_len)
train, test = dataet[0:train_end,:],dataset[train_end:data_len,:]

# Normalize features
scaler_all = StandardScaler(copy = False)
scaler_all.fit_transform(train[:,0:-1])
scaler_last = StandardScaler(copy = False)
scaler_last.fit_transform(train[:,-1].reshape(train.shape[0],1))
scaler_all.transform(test[:,0:-1])
scaler_last.transform(test[:,-1].reshape(test.shape[0],1))

# Convert an array of values into a dataset matrix
# look_back is the number of previous time steps to use as input to predict
# time_steps is the number of days ahead to forecast
# Set default values as 1
def create_dataset(dataset, look_back = 1, time_steps = 1):
    dataX, DataY = [], []
    for i in range(len(dataset)-look_back-time_steps-1):
        a = dataset[i:(i+look_back),0:-1]
        b = dataset[(i+look_back+time_steps),-1]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX),np.array(dataY)

# create test and train datasets as into X = t and Y = t + look_back + time_steps
look_back = 5
time_steps = 1
x_train, y_train = create_dataset(train,look_back,time_steps)
x_test, y_test = create_dataset(test,look_back,time_steps)

# reshape input to be [samples, look_back, features]
x_train = np.reshape(x_train, (x_train.shape[0], look_back, x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], look_back, x_test.shape[2]))
y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))

# LSTM model
model = Sequential()
model.add(LSTM(1000, recurrent_activation = "hard_sigmoid", input_shape = (look_back, x_train.shape[2]), activation = "tanh", return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(100, recurrent_activation = "hard_sigmoid", activation = "tanh", return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(activation = "linear", units = 1))

# sgd = optimizers.SGD(lr = 1.00, decay = 1e-2, momentum = 0.9, nesterov = True)
model.compile(loss = "mean_square_error", optimizer = "adadelta")
history = model.fit(x_train, y_train, batch_size = 60, epochs = 50, shuffle = False, validation_split = 0.05)

print(model.summary())

# Plot history
plt.plot(history.history['loss'], label = "training loss")
plt.plot(hisotry.history['val_loss'], label = "validation loss")
plt.show()

score_train = model.evaluate(x_train, y_train, batch_size = 1)
score_test = model.evaluate(x_test, y_test, batch_size = 1)
print("in train MSE = ", round(score_train,4))
print("in tets MSE = ", round(score_test,4))

y_pred = model.predict(x_test)
y_pred = scaler_last.inverse_transform(y_pred)
y_test = scaler_last.inverse_transform(y_test)
y_test = y_test.reshape(y_pred.shape)

plt.figure()
plt.plot(y_pred, label = "predictions")
plt.plot(y_test, label = "actual")
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, ncol = 2)
plt.show()

export_result = np.concatenate((y_test, y_pred), axis = 1)
np.savetxt("Result.csv", export_result, delimiter = ',', fmt = '%s')

model.save('LSTM_Model.h5')
