from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import data_preparation as dp

N_FEATURES=22
MB_SIZE = 6793
STEP_PER_EPOCH = 6
LOOK_BACK = 70
LSTM_NEURONS = 50
DENSE_NEURONS = 1
EPOCHS = 15
 
# creo un nuovo dataset come input per la rete
def create_dataset(data, n_in=1):
	n_vars = data.shape[1] #n° features
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	cols.append(df.shift(-i))
	names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	return agg


# load dataset
dataset = pd.read_csv("./DataSet/voli_2014_2.csv", header=0, delimiter=';')
#Elimino le ultime features inutili
useless = ['Unnamed: 109', 'Unnamed: 0', 'Unnamed: 1']   #???????????
dp.clean_ds(dataset, useless)

# integer encode direction
#values[:,4] = LabelEncoder().fit_transform(values[:,4]) #!!!!
columns=["OriginState", "DestState"]
for col in columns:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])
dataset.to_csv("voli_finale_2014_2.csv",sep = ';',decimal = ',')

values = dataset.values
# ensure all data is float
values = values.astype('float64')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = create_dataset(scaled, LOOK_BACK, 1)
print(reframed.shape)



#--------------------------------------------------------------------------------------------------------------------------------------


# drop columns we don't want to predict - quelle che aggiunge alla creazione del dataset
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())


# split into train and test sets
values = reframed.values
train, test = dp.split_data(values)



# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape, len(train_X), train_y.shape)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], LOOK_BACK, train_X.shape[1]))     #train_X.shape[1] oppure 1 oppure n° features ????
test_X = test_X.reshape((test_X.shape[0], LOOK_BACK, test_X.shape[1]))         #train_X.shape[1] oppure 1 oppure n° features ????

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

