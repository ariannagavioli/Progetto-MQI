from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import data_preparation as dp

N_FEATURES=21
LOOK_BACK = 70
LSTM_NEURONS = 50
DENSE_NEURONS = 1
EPOCHS = 15
PROVA=1
 
# creo un nuovo dataset come input per la rete
def create_dataset(data, col, n_in=1):
	n_vars = data.shape[1] #n° features
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [(col[j]+'(t-%d)' % (i)) for j in range(n_vars)]
	# forecast time t
	cols.append(df)
	names += [(col[j]+'(t)') for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	return agg


# load dataset
dataset = pd.read_csv("./DataSet/voli_2014_2.csv", header=0, delimiter=';')
dataset = dataset.iloc[:10000, :]
#Elimino le ultime features inutili
useless = ['Unnamed: 109', 'Unnamed: 0', 'Unnamed: 1', 'FlightDate']
dp.clean_ds(dataset, useless)

# integer encode direction
#values[:,4] = LabelEncoder().fit_transform(values[:,4]) #!!!!
columns=["OriginState", "DestState"]
for col in columns:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])
#dataset.to_csv("voli_finale_2014_2.csv",sep = ';',decimal = ',')   !!!!!

values = dataset.values
# ensure all data is float
values = values.astype('float64')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = create_dataset(scaled, dataset.columns, LOOK_BACK)
print(reframed.shape)

# split into train and test sets
#values = reframed.values
'''
Divisione sequenziale:
n_train_hours = 365 * 70 
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
'''

train, test = dp.split_data2(reframed.iloc[70:])

to_predict= dp.obtain_index(dataset, "ArrDelayMinutes")
print("indice target=")
print (to_predict)


# drop columns we don't want to predict - quelle che aggiunge alla creazione del dataset
for i in range (0, N_FEATURES):
    if (i<to_predict): train.drop(train.columns[(LOOK_BACK*N_FEATURES)], axis=1, inplace=True)
    if (i>to_predict): train.drop(train.columns[(LOOK_BACK*N_FEATURES)+1], axis=1, inplace=True)
print(train.head())


# drop columns we don't want to predict - quelle che aggiunge alla creazione del dataset
for i in range (0, N_FEATURES):
    if (i<to_predict): test.drop(test.columns[(LOOK_BACK*N_FEATURES)], axis=1, inplace=True)
    if (i>to_predict): test.drop(test.columns[(LOOK_BACK*N_FEATURES)+1], axis=1, inplace=True)
print(test.head())


train = train.values
test = test.values

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], LOOK_BACK, N_FEATURES))     #train_X.shape[1] e n° features non sono uguali????
test_X = test_X.reshape((test_X.shape[0], LOOK_BACK, N_FEATURES))         #train_X.shape[1] e n° features non sono uguali????

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(LSTM_NEURONS, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(Dense(DENSE_NEURONS,kernel_initializer='random_uniform',bias_initializer='zeros'))
#SGD = Stochastic Gradient Descent
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
# fit network
#INIZIALMENTE batch_size=72, POI MESSO QUELLO DI DEFAULT (32)
history = model.fit(train_X, train_y, epochs=EPOCHS, validation_data=(test_X, test_y), verbose=1, shuffle=False)
#PER SALVARE
model.save("LSTM_multivariate_prova"+str(PROVA)+".h5")
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -(N_FEATURES-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -(N_FEATURES-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

