import pandas as pd
from sklearn.utils import shuffle

DATA_PATH = "./DataSet/"
REPOSITORY = "/On_Time_On_Time_Performance_"
CSV_PATH = "/On_Time_On_Time_Performance_"
useless1 = ["Quarter","UniqueCarrier","Carrier","TailNum","FlightNum","OriginAirportSeqID","Origin","OriginCityMarketID","OriginCityName","OriginStateFips","OriginStateName","OriginWac",
            "DestAirportSeqID","Dest","DestCityMarketID","DestCityName","DestStateFips","DestStateName","DestWac","DepDelay","DepDel15","DepartureDelayGroups","DepTimeBlk","TaxiOut","WheelsOff",
            "WheelsOn","TaxiIn","ArrDelay","ArrDel15","ArrivalDelayGroups","ArrTimeBlk","Cancelled","CancellationCode","Diverted",
            "CRSElapsedTime","ActualElapsedTime","AirTime","Flights","Distance","FirstDepTime","TotalAddGTime","LongestAddGTime",
            "DivAirportLandings","DivReachedDest","DivActualElapsedTime","DivArrDelay","DivDistance","Div1Airport","Div1AirportID","Div1AirportSeqID","Div1WheelsOn","Div1TotalGTime",
            "Div1LongestGTime","Div1WheelsOff","Div1TailNum","Div2Airport","Div2AirportID","Div2AirportSeqID","Div2WheelsOn","Div2TotalGTime","Div2LongestGTime","Div2WheelsOff","Div2TailNum",
            "Div3Airport","Div3AirportID","Div3AirportSeqID","Div3WheelsOn","Div3TotalGTime","Div3LongestGTime","Div3WheelsOff","Div3TailNum","Div4Airport","Div4AirportID","Div4AirportSeqID",
            "Div4WheelsOn","Div4TotalGTime","Div4LongestGTime","Div4WheelsOff","Div4TailNum","Div5Airport","Div5AirportID","Div5AirportSeqID","Div5WheelsOn","Div5TotalGTime","Div5LongestGTime",
            "Div5WheelsOff","Div5TailNum"]

#clean_ds toglie dal dataframe tutte le features contenute nel vettore vet
def clean_ds(dataframe, vet=useless1):
    dataframe.dropna(subset=["Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate", "AirlineID", "OriginAirportID", "OriginState",
                             "DestAirportID", "DestState", "CRSDepTime", "DepTime", "DepDelayMinutes", "CRSArrTime", "ArrTime",
                             "ArrDelayMinutes"], inplace=True)
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

# create_ds unisce tutti i mesi dell'anno
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

#unify_ds unisce gli anni ridotti ad una sola tratta e crea un nuovo file || crea un file per ogni anno
def unify_ds():
    '''#UNICA TRATTA, TUTTI GLI ANNI
    anni_puliti = {}
    for year in range(0,4):
        anni_puliti[year] = create_ds(year+2014)
        #in ris ci sono solo i voli da NY a CA
        anni_puliti[year] = anni_puliti[year][(anni_puliti[year]['OriginState'] == 'NY') & (anni_puliti[year]['DestState'] == 'CA')]
        anni_puliti[year] = anni_puliti[year].sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True])
    tutti_anni = pd.concat(anni_puliti)
    tutti_anni.to_csv("voli_unicatratta_2.csv",sep = ';',decimal = ',')
    '''
    #COMPLETO UNICO ANNO
    for year in range(0,4):
        anno_pulito = create_ds(year+2014)
        anno_pulito= anno_pulito.sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True])
        anno_pulito.to_csv("voli_"+str(year+2014)+"_2.csv",sep = ';',decimal = ',')
    return

#obtain_index restituisce l'indice della colonna di nome index_name
def obtain_index(data_frame, index_name):
    name_list = list(data_frame.columns.values)
    for i in range(len(name_list)):
        if(name_list[i]==index_name):
            return i