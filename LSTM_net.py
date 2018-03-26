import numpy
#import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

LOOK_BACK = 70
DENSE_BLOCKS = 1
LSTM_BLOCKS = 4
EPOCHS = 1
PROVA = 1
PATH_TRAIN = "./DataSet/train_set.csv"
PATH_TEST = "./DataSet/test_set.csv"

train = pd.read_csv(PATH_TRAIN,delimiter=';')
test = pd.read_csv(PATH_TEST,delimiter=';')

train.pop('FlightDate')
train.pop('Unnamed: 109')
test.pop('FlightDate')
test.pop('Unnamed: 109')
train.pop('CancellationCode')
test.pop('CancellationCode')

train.astype(dtype = 'float64')
test.astype(dtype = 'float64')

#Normalizzo i valori nei set
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

'''def delay_finder(dataset):
    print (dataset.columns)
    for i in range (dataset.shape[1]):
        print(dataset.iloc[0,i])
        print('\n')
        if(dataset[0,i]=='ArrDelayMinutes'): return i
    return -1'''

#Creo un nuovo dataset con timestamp consecutivi su una stessa riga
#Di fatto avrò i voli precedenti al volo i in voliPrecedenti[i], il ritardo da
#predire del volo i in ritardiEffettivi[i]
def create_dataset(dataset, look_back=LOOK_BACK):
    #nColonne = dataset.shape[1]
    #index_delay = delay_finder(dataset)
    #print (index_delay)
    #print (dataset.columns)
    print ("\n\n\n\n\n")
    voliPrecedenti, ritardiEffettivi = [],[]
    print (dataset.shape[0])
    print ("\n\n\n\n\n")
    print (dataset.shape[1])
    print ("\n\n\n\n\n")
    for i in range(dataset.shape[0]-look_back):
        x = dataset[i:(i+look_back),:]
        voliPrecedenti.append(x)
        ritardiEffettivi.append(dataset[i+look_back,15])
    return numpy.array(voliPrecedenti), numpy.array(ritardiEffettivi)

voliPrecedenti, ritardiEffettivi = create_dataset(train)
testVoliPrecedenti, testRitardiEffettivi = create_dataset(test)

#Trasformo i dati nella forma [campioni,time step, feature]
#voliPrecedenti.shape[0] => numero righe = numeor di campioni
#voliPrecedenti.shape[1] => numero colonne = numero di feature
#time step = 1 perché facciamo scorrere la finestra temporale con un passo = 1
print("voliPrecedenti.shape[2]:")
print(voliPrecedenti.shape[2])
print("\n\n\n\n\n")
print("voliPrecedenti.shape[1]:")
print(voliPrecedenti.shape[1])
print("\n\n\n\n\n")
print("voliPrecedenti.shape[0]:")
print(voliPrecedenti.shape[0])
print("\n\n\n\n\n")
voliPrecedenti = numpy.reshape(voliPrecedenti, (voliPrecedenti.shape[0],LOOK_BACK,voliPrecedenti.shape[2]))
testVoliPrecedenti = numpy.reshape(testVoliPrecedenti, (testVoliPrecedenti.shape[0],LOOK_BACK,testVoliPrecedenti.shape[2]))

#Creo la rete e la alleno
model = Sequential()
#EVENTUALMENTE INIZIALIZZARE I PESI
#????? Proviamo a scambiare
model.add(LSTM(LSTM_BLOCKS, input_shape = (LOOK_BACK,voliPrecedenti.shape[2])))
model.add(Dense(DENSE_BLOCKS))
#SGD = Stochastic Gradient Descent
model.compile(optimizer = 'SGD', loss = 'mean_squared_error')
#Batch size -> ogni quanto aggiorniamo il gradient descent, batch size di default 32
model.fit(voliPrecedenti, ritardiEffettivi, epochs = EPOCHS, batch_size=1, verbose=1,
          validation_split=0.2)
model.save("LSTN_prova"+str(PROVA)+".h5")

#Prediciamo i ritardi
ritardiPredetti = model.predict(ritardiEffettivi)
testPredict = model.predict(testRitardiEffettivi)
#Riportiamo i dati normalizzati tra (0,1) nelle unità originali
ritardiPredetti = scaler.inverse_transform(ritardiPredetti)
ritardiEffettivi = scaler.inverse_transform([ritardiEffettivi])
testPredict = scaler.inverse_transform(testPredict)
testRitardiEffettivi = scaler.inverse_transform([testRitardiEffettivi])
#Calcolo l'errore con lo scarto quadratico medio
trainError = math.sqrt(mean_squared_error(ritardiEffettivi[0],ritardiPredetti.iloc[:,0])) 
print("Errore di train: %.2f RMSE" % (trainError))
testError = math.sqrt(mean_squared_error(testRitardiEffettivi[0],testPredict.iloc[:,0])) 
print("Errore di test: %.2f RMSE" % (testError))