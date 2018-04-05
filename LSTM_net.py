import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from data_preparation import obtain_index

#MACROS
MB_SIZE = 6793          #dimensione del minibatch
STEP_PER_EPOCH = 6      #numero dei minibatch nel dataset
LOOK_BACK = 70          #numero dei voli nella finestra temporale precedente
LSTM_NEURONS = 50       #numero neuroni LSTM
DENSE_NEURONS = 1       #numero neuroni Dense
EPOCHS = 15             #numero di epoche per l'allenamento
PROVA = 1               #numero di prova per salvare l'allenamento

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

#Il dataset viene castato a float
train.astype(dtype = 'float64')
test.astype(dtype = 'float64')

#Creo un nuovo dataset con timestamp consecutivi su una stessa riga
#Di fatto avrò i voli precedenti al volo i in voliPrecedenti[i], il ritardo da
#predire del volo i in ritardiEffettivi[i]
def create_dataset(dataset, look_back=LOOK_BACK):
    #Inizializzo voliPrecedenti a zero con dimensioni (numero voli da predire, time step precedenti, numero di feature)
    #Inizializzo ritardiEffettivi a zero con dimensioni (numero voli da predire, 1)
    voliPrecedenti = numpy.zeros((dataset.shape[0]-look_back,look_back+1,dataset.shape[1]),dtype=numpy.float)
    ritardiEffettivi = numpy.zeros((dataset.shape[0]-look_back,1),dtype=numpy.float)
    for i in range(dataset.shape[0]-look_back):
        if(i%2): print(i)
        x = dataset[i:(i+look_back+1)]
        #x[i+look_back] contiene le feature del volo di cui vogliamo predire il ritardo
        #impostiamo i valori DepDelayMinutes, ArrDelayMinutes, ... alla media dei ritardi della 
        #finestra precedente (i:i+look_back), in modo da non influenzare la predizione del ritardo
        dep_delay = obtain_index(dataset,"DepDelayMinutes")
        arr_delay = obtain_index(dataset,"ArrDelayMinutes")
        carrier_delay = obtain_index(dataset,"CarrierDelay")
        weather_delay = obtain_index(dataset,"WeatherDelay")
        nas_delay = obtain_index(dataset,"NASDelay")
        security_delay = obtain_index(dataset,"SecurityDelay")
        lateairc_delay = obtain_index(dataset,"LateAircraftDelay")
        arr_time = obtain_index(dataset,"ArrTime")
        crs_arr_time = obtain_index(dataset,"CRSArrTime")
        x.iloc[look_back,dep_delay] = average(x.iloc[0:(look_back)],dep_delay)
        x.iloc[look_back,arr_delay] = average(x.iloc[0:(look_back)],arr_delay)
        x.iloc[look_back,carrier_delay] = average(x.iloc[0:(look_back)],carrier_delay)
        x.iloc[look_back,weather_delay] = average(x.iloc[0:(look_back)],weather_delay)
        x.iloc[look_back,nas_delay] = average(x.iloc[0:(look_back)],nas_delay)
        x.iloc[look_back,security_delay] = average(x.iloc[0:(look_back)],security_delay)
        x.iloc[look_back,lateairc_delay] = average(x.iloc[0:(look_back)],lateairc_delay)
        #imposto l'orario di arrivo all'orario previsto di arrivo + minuti di ritardo previsti
        #in modo naif (media) poiché non abbiamo informazioni sull'effettivo ritardo
        x.iloc[look_back,arr_time] = x.iloc[look_back,crs_arr_time] + x.iloc[look_back,arr_delay]
        voliPrecedenti[i] = x
        ritardiEffettivi[i] = dataset.iloc[i+look_back,arr_delay]
    return voliPrecedenti,ritardiEffettivi

#Definisco una funzione che ritorna la media di una colonna di un dataframe
def average(dataset,index):
    return numpy.mean(dataset.iloc[index])

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

voliPrecedenti.to_csv("voliPrecedenti.csv",sep=';',decimal=',')
ritardiEffettivi.to_csv("ritardiEffettivi.csv",sep=';',decimal=',')
testVoliPrecedenti.to_csv("testVoliPrecedenti.csv",sep=';',decimal=',')
testRitardiEffettivi.to_csv("testRitardiEffettivi.csv",sep=';',decimal=',')


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

#COME PRIMO PARAMETRO I LAYER VOGLIONO IL NUMERO DI NEURONI
model.add(LSTM(LSTM_NEURONS, input_shape = (LOOK_BACK,voliPrecedenti.shape[2]),
                                            kernel_initializer='random_uniform',bias_initializer='zeros',
                                            activation = 'sigmoid'))
model.add(Dense(DENSE_NEURONS,kernel_initializer='random_uniform',bias_initializer='zeros'))
#SGD = Stochastic Gradient Descent
model.compile(optimizer = 'SGD', loss = 'mean_squared_error')
minibatch_size = MB_SIZE
#Batch size -> ogni quanto aggiorniamo il gradient descent, batch size di default 32
#model.fit_generator(generator(minibatch_size), steps_per_epoch = STEP_PER_EPOCH, epochs = EPOCHS, verbose=1,
#                    use_multiprocessing = True)
history = model.fit(voliPrecedenti, ritardiEffettivi, epochs = EPOCHS, verbose = 1, validation_split = 0.2)
model.save("LSTM_prova"+str(PROVA)+".h5")

#QUESTO L'HA SCRITTO LEO questa cosa non funziona, potresti provare a seguire
#il flusso delle operazioni riportate qui: https://datascience.stackexchange.com/questions/16548/minmaxscaler-broadcast-shapes/16741#16741
#il top sarebbe fit_trasformare i dati prima di fare il fit quindi prova a riscrivere la create

#Plot allenamento
#grafico l'andamento di loss e val_loss allo scorrere delle epoche
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
#plt.savefig("plot1.png")
plt.show()

#Prediciamo i ritardi
ritardiPredetti = model.predict(voliPrecedenti,verbose=1)
testPredict = model.predict(testVoliPrecedenti,verbose=1)
#Riportiamo i dati normalizzati tra (0,1) nelle unità originali
ritardiPredetti = scaler2.inverse_transform(ritardiPredetti)
ritardiEffettivi = scaler2.inverse_transform(ritardiEffettivi)
testPredict = scaler2.inverse_transform(testPredict)
testRitardiEffettivi = scaler2.inverse_transform(testRitardiEffettivi)

#Plot predizione finale
plt.plot(testPredict, label='predict')
plt.legend()
#plt.savefig("plot2.png")
plt.show()
#Plot ritardi effettivi
plt.plot(testRitardiEffettivi, label='true')
plt.legend()
#plt.savefig("plot3.png")
plt.show()

#Calcolo l'errore con lo scarto quadratico medio
trainError = math.sqrt(mean_squared_error(ritardiEffettivi,ritardiPredetti)) 
print("Errore di train: %.2f RMSE" % (trainError))
testError = math.sqrt(mean_squared_error(testRitardiEffettivi,testPredict)) 
print("Errore di test: %.2f RMSE" % (testError))