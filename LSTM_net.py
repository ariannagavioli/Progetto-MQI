import numpy
#import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

LOOK_BACK = 70
DENSE_BLOCKS = 1
LSTM_BLOCKS = 4
EPOCHS = 100
PATH_TRAIN = "./DataSet/train_set.csv"
PATH_TEST = "./DataSet/test_set.csv"

train = pd.read_csv(PATH_TRAIN,delimiter=';')
test = pd.read_csv(PATH_TEST,delimiter=';')

train.pop('FlightDate')
test.pop('FlightDate')

#Normalizzo i valori nei set
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

#Creo un nuovo dataset con timestamp consecutivi su una stessa riga
#Di fatto avrò i voli precedenti al volo i in voliPrecedenti[i], il ritardo da
#predire del volo i in ritardiEffettivi[i]
def create_dataset(dataset, look_back=LOOK_BACK):
    nColonne = len(dataset.columns)
    voliPrecedenti, ritardiEffettivi = [],[]
    for i in range(len(dataset)-look_back-1):
        x = dataset[i:(i+look_back),nColonne]
        voliPrecedenti.append(x)
        ritardiEffettivi.append(dataset[i+look_back,nColonne]['ArrDelayMinutes'])
    return numpy.array(voliPrecedenti), numpy.array(ritardiEffettivi)

voliPrecedenti, ritardiEffettivi = create_dataset(train)
testVoliPrecedenti, testRitardiEffettivi = create_dataset(test)

#Trasformo i dati nella forma [campioni,time step, feature]
#voliPrecedenti.shape[0] => numero righe = numeor di campioni
#voliPrecedenti.shape[1] => numero colonne = numero di feature
#time step = 1 perché facciamo scorrere la finestra temporale con un passo = 1
voliPrecedenti = numpy.reshape(voliPrecedenti, (voliPrecedenti.shape[0],1,voliPrecedenti.shape[1]))
testVoliPrecedenti = numpy.reshape(testVoliPrecedenti, (testVoliPrecedenti.shape[0],1,testVoliPrecedenti.shape[1]))

#Creo la rete e la alleno
model = Sequential()
#EVENTUALMENTE INIZIALIZZARE I PESI
model.add(LSTM(LSTM_BLOCKS, input_shape = (voliPrecedenti.shape[1],LOOK_BACK)))
model.add(Dense(DENSE_BLOCKS))
#SGD = Stochastic Gradient Descent
model.compile(optimizer = 'SGD', loss = 'mean_squared_error')
#Batch size -> ogni quanto aggiorniamo il gradient descent, batch size di default 32
model.fit(voliPrecedenti, ritardiEffettivi, epochs = EPOCHS, batch_size=1, verbose=1,
          validation_split=0.2)

#Prediciamo i ritardi
ritardiPredetti = model.predict(ritardiEffettivi)
testPredict = model.predict(testRitardiEffettivi)
#Riportiamo i dati normalizzati tra (0,1) nelle unità originali
ritardiPredetti = scaler.inverse_transform(ritardiPredetti)
ritardiEffettivi = scaler.inverse_transform([ritardiEffettivi])
testPredict = scaler.inverse_transform(testPredict)
testRitardiEffettivi = scaler.inverse_transform([testRitardiEffettivi])
#Calcolo l'errore con lo scarto quadratico medio
trainError = math.sqrt(mean_squared_error(ritardiEffettivi[0],ritardiPredetti[:,0])) 
print("Errore di train: %.2f RMSE" % (trainError))
testError = math.sqrt(mean_squared_error(testRitardiEffettivi[0],testPredict[:,0])) 
print("Errore di test: %.2f RMSE" % (testError))