import pandas as pd

DATA_PATH = "./DataSet/On_Time_On_Time_Performance_2017_"
CSV_PATH = "/On_Time_On_Time_Performance_2017_" 

def creaDS():
    csvs = {}
    for i in range(1,3):
        DATA_PATH_i = DATA_PATH + str(i)
        print(DATA_PATH_i)
        CSV_PATH_i = CSV_PATH + str(i)
        print(CSV_PATH_i)
        csvs[i-1] = pd.read_csv(DATA_PATH_i + CSV_PATH_i+".csv")
    result = pd.concat(csvs)
    return result

a = creaDS()
a.to_csv("Voli2017.csv",sep = ';',decimal = ',')
