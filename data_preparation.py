'''IL FILE GENERATO "UNICA_TRATTA_PULITA.CSV" È UN FILE CSV CHE CONTIENE SOLO I VOLI 
NY->CA DOVE SONO STATE ELIMINATE LE COLONNE DEL NOME DELLO
STATO DI ARRIVO E PARTENZA, POICHÈ NOTE'''
import pandas as pd

PATH1 = "/home/luca/Scrivania/PROGETTO/On_Time_On_Time_Performance_2015_1.csv"
useless1 = ["Quarter","Year","Month","DayofMonth","DayOfWeek","UniqueCarrier","Carrier","TailNum","FlightNum","OriginAirportSeqID","OriginCityMarketID","OriginCityName","OriginStateFips","OriginStateName","OriginWac","DestAirportSeqID",
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
def clean_e_save_ds(dataframe, vet, name):
    dataframe.fillna(0, inplace=True)
    for i in range(0,len(vet)):
         del dataframe[vet[i]]
    dataframe.to_csv(name, sep = ';', decimal = ',')
 
#leggo il csv e lo salvo in un dataframe
df = pd.read_csv(PATH1,dtype=object)
clean_e_save_ds(df, useless1, "voli_oraperora.csv")
#in ris ci sono solo i voli da NY a CA
ris = df[(df['OriginState'] == 'NY') & (df['DestState'] == 'CA')]
ris.to_csv("voli2015_unicatratta.csv",sep = ';',decimal = ',')
clean_e_save_ds(ris, useless2, "unica_tratta_pulita.csv")

ris["date_time"]=ris["FlightDate"].map(str)+ris["DepTime"].map(str)
ris.pop("Unnamed: 109")
ris.sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True],inplace = True)
ris.to_csv("perfetto_plot.csv",sep = ';',decimal = ',')