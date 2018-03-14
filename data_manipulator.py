import pandas as pd

DATA_PATH = "./DataSet/"
REPOSITORY = "/On_Time_On_Time_Performance_"
CSV_PATH = "/On_Time_On_Time_Performance_" 
year = 2014

def creaDS():
    csvs = {}
    for month in range(1,12):
        #for year in range(2014,2017):
            DATA_PATH_i = DATA_PATH + str(year) + REPOSITORY + str(year) + "_" + str(month)
            print(DATA_PATH_i)
            CSV_PATH_i = CSV_PATH + str(year) + "_" +  str(month)
            print(CSV_PATH_i)
            csvs[month-1] = pd.read_csv(DATA_PATH_i + CSV_PATH_i+".csv",dtype=object)
    result = pd.concat(csvs)
    return result

dati = creaDS()

#dati.sort_values(by="DayOfMonth", inplace="True")
useless = ["FlightDate","UniqueCarrier","Carrier","TailNum","FlightNum","OriginAirportSeqID","OriginCityMarketID","OriginCityName","OriginStateFips","OriginStateName","OriginWac","DestAirportSeqID",
"DestCityMarketID","DestCityName","DestStateFips","DestStateName","DestWac","DepTime","DepDelay","DepDel15","DepartureDelayGroups","DepTimeBlk","TaxiOut","WheelsOff","WheelsOn","TaxiIn",
"ArrTime","ArrDelay","ArrDel15","ArrivalDelayGroups","ArrTimeBlk","CRSElapsedTime","ActualElapsedTime","AirTime","Flights","Distance","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay","FirstDepTime",
"TotalAddGTime","LongestAddGTime","DivAirportLandings","DivReachedDest","DivActualElapsedTime","DivArrDelay","DivDistance","Div1Airport","Div1AirportID","Div1AirportSeqID","Div1WheelsOn","Div1TotalGTime","Div1LongestGTime",
"Div1WheelsOff","Div1TailNum","Div2Airport","Div2AirportID","Div2AirportSeqID","Div2WheelsOn","Div2TotalGTime","Div2LongestGTime",
"Div2WheelsOff","Div2TailNum","Div3Airport","Div3AirportID","Div3AirportSeqID","Div3WheelsOn","Div3TotalGTime","Div3LongestGTime",
"Div3WheelsOff","Div3TailNum","Div4Airport","Div4AirportID","Div4AirportSeqID","Div4WheelsOn","Div4TotalGTime","Div4LongestGTime",
"Div4WheelsOff","Div4TailNum","Div5Airport","Div5AirportID","Div5AirportSeqID","Div5WheelsOn","Div5TotalGTime","Div5LongestGTime",
"Div5WheelsOff","Div5TailNum"]

for i in range(0,len(useless)):
	del dati[useless[i]]


dati.pop('Unnamed: 109')
dati.to_csv("Voli"+str(year)+".csv",sep = ';',decimal = ',')

