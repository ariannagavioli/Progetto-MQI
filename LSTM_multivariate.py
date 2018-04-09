from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import data_preparation as dp

LOOK_BACK = 70
LSTM_NEURONS = 64
DENSE_NEURONS = 1
EPOCHS = 25
PROVA = 24
BATCH_SIZE = 32
 
#creiamo un nuovo dataset come input per la rete
def create_dataset(data, col, n_in=1):
    n_vars = data.shape[1] #n° features
    df = pd.DataFrame(data)
    cols, names = list(), list()
    #creiamo le sequenze (t-look_back, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(col[j]+'(t-%d)' % (i)) for j in range(n_vars)]
    #aggiungiamo la colonna 
    cols.append(df)
    names += [(col[j]+'(t)') for j in range(n_vars)]
    #concateniamo tutto insieme
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg


#carichiamo il dataset
dataset = pd.read_csv("./DataSet/voli_unicatratta_2.csv", header=0, delimiter=';')
#eliminiamo le ultime features inutili
useless = ['Unnamed: 109', 'Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'FlightDate']
dp.clean_ds(dataset, useless)

#utilizziamo un label encoder per trasformare i codici degli stati di origine e
#destinazione in un intero per darlo in pasto alla rete
columns=["OriginState", "DestState"]
for col in columns:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])

values = dataset.values
#ci assicuriamo che tutti i dati siano float
values = values.astype('float64')
#normalizziamo i dati tra -1 e 1 poiché utilizziamo come funzione di attivazione
#nello strato LSTM la tanh
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
#mettiamo i dati in forma features(t-look_back), features(t-look_back+1), ..., features(t)
reframed = create_dataset(scaled, dataset.columns, LOOK_BACK)

#eliminiamo al tempo t tutte le feature che non ci interessano, lasciando unicamente l'ArrDelayMinutes
to_predict= dp.obtain_index(dataset, "ArrDelayMinutes")
n_features = reframed.shape[1]/(LOOK_BACK+1)
for i in range (0, n_features):
    if (i<to_predict): reframed.drop(reframed.columns[(LOOK_BACK*n_features)], axis=1, inplace=True)
    if (i>to_predict): reframed.drop(reframed.columns[(LOOK_BACK*n_features)+1], axis=1, inplace=True)

values = reframed.values

#divisione sequenziale in train e test:
train_size = int(len(values)*0.8)
train = values[LOOK_BACK:train_size, :]
test = values[train_size:, :]

#split delle sequenze in input e output della rete (voli precedenti e ritardo effettivo)
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#modifichiamo l'input della rete in modo che sia tridimensionale,
#nella forma [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], LOOK_BACK, n_features))     #train_X.shape[1] e n° features non sono uguali????
test_X = test_X.reshape((test_X.shape[0], LOOK_BACK, n_features))         #train_X.shape[1] e n° features non sono uguali????

#costruiamo la rete
model = Sequential()
model.add(LSTM(LSTM_NEURONS, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='random_uniform',bias_initializer='zeros'))#,activation = 'sigmoid'))
model.add(Dense(DENSE_NEURONS,kernel_initializer='random_uniform',bias_initializer='zeros',activation='linear'))
#come ottimizzatore usiamo Adam
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
#fit della rete
history = model.fit(train_X, train_y, epochs=EPOCHS, validation_split = 0.2, verbose=1, shuffle=True, batch_size = BATCH_SIZE)
#salviamo la rete
model.save("LSTM_multivariate_prova"+str(PROVA)+".h5")
#plot dell'evoluzione dei parametri di loss durante l'allenamento
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('plot_validation_tot_prova'+str(PROVA)+'.png')
plt.show()


#proviamo a predire i ritardi del test set
yhat = model.predict(test_X)
#vogliamo riportare i dati nella loro scala originale
#ricordiamo che lo scaler.inverse cercherà di riportare i dati nelle dimensioni
#originali, per cui dobbiamo passargli dei dati coerenti
#riportiamo pertanto test_X alle sue dimensioni originali
test_X = test_X.reshape((test_X.shape[0], LOOK_BACK*n_features))
#invertiamo lo scaling in modo da riportare la previsione fatta dalla rete
#nella scala dei ritardi originari
inv_yhat = np.concatenate((test_X[:,0:to_predict],yhat,test_X[:,to_predict+1:scaled.shape[1]]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,to_predict]
#invertiamo lo scaling per riportare i ritardi effettivi alla loro scala originale
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:,0:to_predict],test_y, test_X[:,to_predict+1:scaled.shape[1]]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,to_predict]

#plottiamo ritardi predetti sul test e ritardi effettivi
plt.plot(inv_yhat, label = 'predict')
plt.plot(inv_y, label = 'true', alpha = 0.5)
plt.legend()
plt.show()
plt.savefig('unicaTratta_prova'+str(PROVA)+'.png')

#calcoliamo il root mean squared error sui ritardi predetti sul test
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


#analogamente, facciamo la stessa cosa sul train set per poterci calcolare
#l'RMSE fatto sui dati del train
yhat_TRAIN = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], LOOK_BACK*n_features))

inv_yhat_TRAIN = np.concatenate((train_X[:,0:to_predict],yhat_TRAIN,train_X[:,to_predict+1:scaled.shape[1]]), axis=1)
inv_yhat_TRAIN = scaler.inverse_transform(inv_yhat_TRAIN)
inv_yhat_TRAIN = inv_yhat_TRAIN[:,to_predict]

train_y = train_y.reshape((len(train_y), 1))
inv_y_TRAIN = np.concatenate((train_X[:,0:to_predict],train_y, train_X[:,to_predict+1:scaled.shape[1]]), axis=1)
inv_y_TRAIN = scaler.inverse_transform(inv_y_TRAIN)
inv_y_TRAIN = inv_y_TRAIN[:,to_predict]

rmse = sqrt(mean_squared_error(inv_y_TRAIN, inv_yhat_TRAIN))
print('Train RMSE: %.3f' % rmse)