from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import data_preparation as dp
#from keras import optimizers

N_FEATURES = 21
LOOK_BACK = 70
LSTM_NEURONS = 64
DENSE_NEURONS = 1
EPOCHS = 10
PROVA = 21
BATCH_SIZE = 32
 
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
dataset = pd.read_csv("./DataSet/voli_unicatratta_2.csv", header=0, delimiter=';')
dataset = dataset.iloc[:, :]
#Elimino le ultime features inutili
useless = ['Unnamed: 109', 'Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'FlightDate']
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


to_predict= dp.obtain_index(dataset, "ArrDelayMinutes")
print("indice target=")
print (to_predict)

# drop columns we don't want to predict - quelle che aggiunge alla creazione del dataset
for i in range (0, N_FEATURES):
    if (i<to_predict): reframed.drop(reframed.columns[(LOOK_BACK*N_FEATURES)], axis=1, inplace=True)
    if (i>to_predict): reframed.drop(reframed.columns[(LOOK_BACK*N_FEATURES)+1], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values

#Divisione sequenziale:
train_size = int(len(values)*0.8)
train = values[LOOK_BACK:train_size, :]
test = values[train_size:, :]


#train, test = dp.split_data2(reframed.iloc[LOOK_BACK:])
'''
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
'''

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
model.add(LSTM(LSTM_NEURONS, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='random_uniform',bias_initializer='zeros'))#,activation = 'sigmoid'))
model.add(Dense(DENSE_NEURONS,kernel_initializer='random_uniform',bias_initializer='zeros',activation='linear'))
#SGD = Stochastic Gradient Descent
#aggiunta per provare modifica learning rate
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
# fit network
#INIZIALMENTE batch_size=72, POI MESSO QUELLO DI DEFAULT (32)
history = model.fit(train_X, train_y, epochs=EPOCHS, validation_split = 0.2, verbose=1, shuffle=True, batch_size = BATCH_SIZE)
#PER SALVARE
model.save("LSTM_multivariate_prova"+str(PROVA)+".h5")
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('plot_validation_tot_prova'+str(PROVA)+'.png')
plt.show()


# make a prediction on test
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], LOOK_BACK*N_FEATURES))

index_delay = dp.obtain_index(dataset,'ArrDelayMinutes')
# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:,0:index_delay],yhat,test_X[:,index_delay+1:scaled.shape[1]]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,index_delay]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:,0:index_delay],test_y, test_X[:,index_delay+1:scaled.shape[1]]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,index_delay]

#plottiamo ritardi predetti sul test e ritardi effettivi
plt.plot(inv_yhat, label = 'predict')
plt.plot(inv_y, label = 'true', alpha = 0.5)
plt.legend()
plt.show()
plt.savefig('unicaTratta_prova'+str(PROVA)+'.png')

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# make a prediction on train to see the rmse
yhat_TRAIN = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], LOOK_BACK*N_FEATURES))

index_delay = dp.obtain_index(dataset,'ArrDelayMinutes')
# invert scaling for forecast
inv_yhat_TRAIN = np.concatenate((train_X[:,0:index_delay],yhat_TRAIN,train_X[:,index_delay+1:scaled.shape[1]]), axis=1)
inv_yhat_TRAIN = scaler.inverse_transform(inv_yhat_TRAIN)
inv_yhat_TRAIN = inv_yhat_TRAIN[:,index_delay]

# invert scaling for actual
train_y = train_y.reshape((len(train_y), 1))
inv_y_TRAIN = np.concatenate((train_X[:,0:index_delay],train_y, train_X[:,index_delay+1:scaled.shape[1]]), axis=1)
inv_y_TRAIN = scaler.inverse_transform(inv_y_TRAIN)
inv_y_TRAIN = inv_y_TRAIN[:,index_delay]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y_TRAIN, inv_yhat_TRAIN))
print('Train RMSE: %.3f' % rmse)