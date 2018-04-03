import numpy as np
import matplotlib.pyplot as plt
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
#non elimino le features inutili perchè prenderò solamente una feature
#il dataset viene castato a float
train.astype(dtype = 'float64')
test.astype(dtype = 'float64')

#Creo un nuovo dataset con timestamp consecutivi su una stessa riga
#Di fatto avrò i voli precedenti al volo i in voliPrecedenti[i], il ritardo da
#predire del volo i in ritardiEffettivi[i]
    
def create_dataset(dataset, look_back=LOOK_BACK):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

#faccio la reshape e prendo solamente la feature DepDelayMinutes
train_values = train["DepDelayMinutes"].values.reshape(-1,1)
test_values = train["DepDelayMinutes"].values.reshape(-1,1)

#Scaler nel caso della sigmoide
scaler_sigmoid1 = MinMaxScaler(feature_range=(0,1))
scaler_sigmoid2 = MinMaxScaler(feature_range=(0,1))
#Scaler nel caso della tanh
scaler_tanh1 = MinMaxScaler(feature_range=(-1,1))
scaler_tanh2 = MinMaxScaler(feature_range=(-1,1))
#riscalo tra -1 e 1
train_set = scaler_tanh1.fit_transform(train_values)
test_set = scaler_tanh1.fit_transform(test_values)

voliPrecedenti, ritardiEffettivi = create_dataset(train_set, LOOK_BACK)
testVoliPrecedenti, testRitardiEffettivi = create_dataset(test_set, LOOK_BACK)
#------------------------------------------------------------------------------
#faccio la reshape del dataset per poterlo dare in pasto alla rete 
voliPrecedenti = np.reshape(voliPrecedenti, (voliPrecedenti.shape[0], 1, voliPrecedenti.shape[1]))
testVoliPrecedenti = np.reshape(testVoliPrecedenti, (testVoliPrecedenti.shape[0], 1, testVoliPrecedenti.shape[1]))

#Creo la rete e la alleno
model = Sequential()
#COME PRIMO PARAMETRO I LAYER VOGLIONO IL NUMERO DI NEURONI
#nelle righe seguenti ci sono due varianti dello strato LSTM
#nella prima viene usata la funzione di attivazione di default che è la tanh
#nella seconda viene usata la sigmoide
#model.add(LSTM(LSTM_NEURONS, input_shape = (LOOK_BACK,voliPrecedenti.shape[2]),kernel_initializer='random_uniform',bias_initializer='zeros'))
#USO LA TANH COME FUNZIONE DI ATTIVAZIONE 
model.add(LSTM(LSTM_NEURONS, input_shape = (voliPrecedenti.shape[1],voliPrecedenti.shape[2]),kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(Dense(DENSE_NEURONS,kernel_initializer='random_uniform',bias_initializer='zeros'))
#SGD = Stochastic Gradient Descent
model.compile(optimizer = 'SGD', loss = 'mean_squared_error')
#minibatch_size = MB_SIZE
#Batch size -> ogni quanto aggiorniamo il gradient descent, batch size di default 32
#model.fit_generator(generator(minibatch_size), steps_per_epoch = STEP_PER_EPOCH, epochs = EPOCHS, verbose=1,
#                    use_multiprocessing = True)
#la funzione fit ritorna un vettore history = ['acc', 'loss', 'val_acc', 'val_loss']
history = model.fit(voliPrecedenti, ritardiEffettivi, epochs = EPOCHS, verbose = 1, validation_split = 0.2)
model.save("LSTM_prova"+str(PROVA)+".h5")

#Plot allenamento
#grafico l'andamento di loss e val_loss allo scorrere delle epoche
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.savefig("lossTrend.png")
plt.show()

#Prediciamo i ritardi
ritardiPredetti = model.predict(voliPrecedenti,verbose=1)
#Plot predizione finale
plt.plot(ritardiPredetti, label='predict')
plt.plot(testRitardiEffettivi, label='predict')
plt.legend()
plt.savefig("plotPrediction.png")
plt.show()

ritardiPredetti_inverse = scaler_tanh1.inverse_transform(ritardiPredetti.reshape(-1,1))
testRitardiEffettivi_inverse = scaler_tanh1.inverse_transform(testRitardiEffettivi.reshape(-1,1))
#Calcolo l'errore con lo scarto quadratico medio

rmse = math.sqrt(mean_squared_error(testRitardiEffettivi_inverse, ritardiPredetti_inverse))
print('Test RMSE: %.3f' % rmse)

plt.plot(ritardiPredetti_inverse, label='predict')
plt.plot(testRitardiEffettivi_inverse, label='real', alpha=0.5)
plt.legend()
plt.savefig("plotFinalPrediction.png")
plt.show()
