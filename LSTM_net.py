import numpy
#import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#macros
MB_SIZE = 6793
STEP_PER_EPOCH = 6
LOOK_BACK = 70
LSTM_NEURONS = 50
DENSE_NEURONS = 1
EPOCHS = 15
PROVA = 1
PATH_TRAIN = "./DataSet/train_subset.csv"
PATH_TEST = "./DataSet/test_subset.csv"
#Importo train e test
train = pd.read_csv(PATH_TRAIN,delimiter=';')
test = pd.read_csv(PATH_TEST,delimiter=';')

#Elimino le ultime features inutili
from data_preparation import clean_ds
useless = ['FlightDate', 'Unnamed: 109', 'Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'CancellationCode']
clean_ds(train, useless)
clean_ds(test, useless)
'''
train.pop('FlightDate')
train.pop('Unnamed: 109')
train.pop('Unnamed: 0')
train.pop('Unnamed: 1')
train.pop('Unnamed: 2')
test.pop('FlightDate')
test.pop('Unnamed: 109')
test.pop('Unnamed: 0')
test.pop('Unnamed: 1')
test.pop('Unnamed: 2')
train.pop('CancellationCode')
test.pop('CancellationCode')
'''
#il dataset viene castato a float
train.astype(dtype = 'float64')
test.astype(dtype = 'float64')

#Creo un nuovo dataset con timestamp consecutivi su una stessa riga
#Di fatto avrò i voli precedenti al volo i in voliPrecedenti[i], il ritardo da
#predire del volo i in ritardiEffettivi[i]
    
def create_dataset(dataset, look_back=LOOK_BACK):
    #Inizializzo voliPrecedenti a zero con dimensioni (numero voli da predire, time step precedenti, numero di feature)
    #Inizializzo ritardiEffettivi a zero con dimensioni (numero voli da predire, 1)
    voliPrecedenti = numpy.zeros((dataset.shape[0]-look_back,look_back,dataset.shape[1]),dtype=numpy.float)
    ritardiEffettivi = numpy.zeros((dataset.shape[0]-look_back,1),dtype=numpy.float)
    for i in range(dataset.shape[0]-look_back):
        x = dataset[i:(i+look_back)]
        voliPrecedenti[i] = x
        ritardiEffettivi[i] = dataset.iloc[i+look_back,15]
    return voliPrecedenti,ritardiEffettivi

#Definisco una funzione generatore di batch per il fit del modello:
def generator(batch_size):
    counter = 0
    while True:
        inputs = voliPrecedenti[counter:counter+batch_size]
        targets = ritardiEffettivi[counter:counter+batch_size]
        counter += 1
        yield (inputs,targets)

voliPrecedenti, ritardiEffettivi = create_dataset(train)
testVoliPrecedenti, testRitardiEffettivi = create_dataset(test)

#Normalizzo i valori nei set
scaler1 = MinMaxScaler(feature_range=(0,1))
scaler2 = MinMaxScaler(feature_range=(0,1))
for i in range (voliPrecedenti.shape[0]):
    voliPrecedenti[i] = scaler1.fit_transform(voliPrecedenti[i])
for i in range (testVoliPrecedenti.shape[0]):
    testVoliPrecedenti[i] = scaler1.fit_transform(testVoliPrecedenti[i])
ritardiEffettivi = scaler2.fit_transform(ritardiEffettivi)
testRitardiEffettivi = scaler2.fit_transform(testRitardiEffettivi)

#Trasformo i dati nella forma [campioni,time step, feature]
#voliPrecedenti.shape[0] => numero righe = numeor di campioni
#voliPrecedenti.shape[1] => numero colonne = numero di feature
#time step = 1 perché facciamo scorrere la finestra temporale con un passo = 1
voliPrecedenti = numpy.reshape(voliPrecedenti, (voliPrecedenti.shape[0],LOOK_BACK,voliPrecedenti.shape[2]))
testVoliPrecedenti = numpy.reshape(testVoliPrecedenti, (testVoliPrecedenti.shape[0],LOOK_BACK,testVoliPrecedenti.shape[2]))

#Creo la rete e la alleno
model = Sequential()
#????? Proviamo a scambiare
#COME PRIMO PARAMETRO I LAYER VOGLIONO IL NUMERO DI NEURONI
model.add(LSTM(LSTM_NEURONS, input_shape = (LOOK_BACK,voliPrecedenti.shape[2]),kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(Dense(DENSE_NEURONS,kernel_initializer='random_uniform',bias_initializer='zeros'))
#SGD = Stochastic Gradient Descent
model.compile(optimizer = 'SGD', loss = 'mean_squared_error')
minibatch_size = MB_SIZE
#Batch size -> ogni quanto aggiorniamo il gradient descent, batch size di default 32
#EVENTUALMENTE METTERE VALIDATION SET
model.fit_generator(generator(minibatch_size), steps_per_epoch = STEP_PER_EPOCH, epochs = EPOCHS, verbose=1,
                    use_multiprocessing = True)
model.save("LSTM_prova"+str(PROVA)+".h5")

#QUESTO L'HA SCRITTO LEO questa cosa non funziona, potresti provare a seguire
#il flusso delle operazioni riportate qui: https://datascience.stackexchange.com/questions/16548/minmaxscaler-broadcast-shapes/16741#16741
#il top sarebbe fit_trasformare i dati prima di fare il fit quindi prova a riscrivere la create

#Prediciamo i ritardi
ritardiPredetti = model.predict(voliPrecedenti,verbose=1)
testPredict = model.predict(testVoliPrecedenti,verbose=1)
#Riportiamo i dati normalizzati tra (0,1) nelle unità originali
ritardiPredetti = scaler2.inverse_transform(ritardiPredetti)
ritardiEffettivi = scaler2.inverse_transform(ritardiEffettivi)
testPredict = scaler2.inverse_transform(testPredict)
testRitardiEffettivi = scaler2.inverse_transform(testRitardiEffettivi)
#Calcolo l'errore con lo scarto quadratico medio
trainError = math.sqrt(mean_squared_error(ritardiEffettivi,ritardiPredetti)) 
print("Errore di train: %.2f RMSE" % (trainError))
testError = math.sqrt(mean_squared_error(testRitardiEffettivi,testPredict)) 
print("Errore di test: %.2f RMSE" % (testError))