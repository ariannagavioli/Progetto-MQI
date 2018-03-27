import pandas as pd
from sklearn.utils import shuffle

PATH_TRAIN = "./DataSet/train_set.csv"
PATH_TEST = "./DataSet/test_set.csv"

def scale_data(data_set, name):
    shuffle_set = shuffle(data_set)
    new_size = int(len(data_set)*0.5)
    new_dataset = shuffle_set.iloc[0:new_size, :]
    new_dataset = new_dataset.sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True])
    new_dataset.to_csv(name+".csv",sep = ';',decimal = '.')
    return

train = pd.read_csv(PATH_TRAIN,delimiter=';')
test = pd.read_csv(PATH_TEST,delimiter=';')
scale_data(train,"train_subset")
scale_data(test,"test_subset")