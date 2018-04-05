'''IL FILE GENERATO "UNICA_TRATTA_PULITA.CSV" È UN FILE CSV CHE CONTIENE SOLO I VOLI 
NY->CA DOVE SONO STATE ELIMINATE LE COLONNE DEL NOME DELLO
STATO DI ARRIVO E PARTENZA, POICHÈ NOTE'''
import pandas as pd
from sklearn.utils import shuffle

DATA_PATH = "./DataSet/"
REPOSITORY = "/On_Time_On_Time_Performance_"
CSV_PATH = "/On_Time_On_Time_Performance_" 
PATH1 = "/home/luca/Scrivania/PROGETTO/On_Time_On_Time_Performance_2015_1.csv"
useless1 = ["Quarter","UniqueCarrier","Carrier","TailNum","FlightNum","OriginAirportSeqID","OriginCityMarketID","OriginCityName","OriginStateFips","OriginStateName","OriginWac","DestAirportSeqID",
"DestCityMarketID","DestCityName","DestStateFips","DestStateName","DestWac","DepDelay","DepDel15","DepartureDelayGroups","DepTimeBlk","TaxiOut","WheelsOff","WheelsOn","TaxiIn",
"ArrTime","ArrDelay","ArrDel15","ArrivalDelayGroups","ArrTimeBlk","CRSElapsedTime","ActualElapsedTime","AirTime","Flights","Distance","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay","FirstDepTime",
"TotalAddGTime","LongestAddGTime","DivAirportLandings","DivReachedDest","DivActualElapsedTime","DivArrDelay","DivDistance","Div1Airport","Div1AirportID","Div1AirportSeqID","Div1WheelsOn","Div1TotalGTime","Div1LongestGTime",
"Div1WheelsOff","Div1TailNum","Div2Airport","Div2AirportID","Div2AirportSeqID","Div2WheelsOn","Div2TotalGTime","Div2LongestGTime",
"Div2WheelsOff","Div2TailNum","Div3Airport","Div3AirportID","Div3AirportSeqID","Div3WheelsOn","Div3TotalGTime","Div3LongestGTime",
"Div3WheelsOff","Div3TailNum","Div4Airport","Div4AirportID","Div4AirportSeqID","Div4WheelsOn","Div4TotalGTime","Div4LongestGTime",
"Div4WheelsOff","Div4TailNum","Div5Airport","Div5AirportID","Div5AirportSeqID","Div5WheelsOn","Div5TotalGTime","Div5LongestGTime",
"Div5WheelsOff","Div5TailNum"]
useless2 = ["Origin","OriginState","Dest","DestState"]
'''
clean_e_save_ds toglie dal dataframe tutte le features contenute nel vettore vet, poi salva il nuovo 
dataframe in un file .csv con nome name
'''
def clean_ds(dataframe, vet=useless1):
    dataframe.fillna(0, inplace=True)
    for i in range(0,len(vet)):
         del dataframe[vet[i]]
    return 
    
def split_data(data_set):
    shuffle_set = shuffle(data_set)
    train_size = int(len(data_set)*0.8)
    train, test = shuffle_set.iloc[0:train_size, :], shuffle_set.iloc[train_size:len(data_set), :]
    train = train.sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True])
    test = test.sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True])
    test.to_csv("test_set.csv",sep = ';',decimal = ',')
    train.to_csv("train_set.csv",sep = ';',decimal = ',')
    return
    
def create_ds(year):
    csvs = {}
    for month in range(1,13):
        DATA_PATH_i = DATA_PATH + str(year) + REPOSITORY + str(year) + "_" + str(month)
        print(DATA_PATH_i)
        CSV_PATH_i = CSV_PATH + str(year) + "_" +  str(month)
        print(CSV_PATH_i)
        csvs[month-1] = pd.read_csv(DATA_PATH_i + CSV_PATH_i+".csv",dtype=object,delimiter=',')
        clean_ds(csvs[month-1],useless1)
    result = pd.concat(csvs)
    return result

def unify_ds():
    #leggo i csv, li pulisco e li salvo 
    anni_puliti = {}
    for year in range(0,4):
        anni_puliti[year] = create_ds(year+2014)
        #in ris ci sono solo i voli da NY a CA
        anni_puliti[year] = anni_puliti[year][(anni_puliti[year]['OriginState'] == 'NY') & (anni_puliti[year]['DestState'] == 'CA')]
        clean_ds(anni_puliti[year], useless2)
        anni_puliti[year].to_csv("voli"+str(year)+"_unicatratta.csv",sep = ';',decimal = ',')
        #creo test e train set
        
    tutti_anni = pd.concat(anni_puliti)
    split_data(tutti_anni)
    return

def obtain_index(data_frame, index_name):
    name_list = list(data_frame.columns.values)
    for i in range(len(name_list)):
        if(name_list[i]=="DepDelayMinutes"):
            return i 
