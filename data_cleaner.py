import pandas as pd

PATH = "/home/ele/Scrivania/gennaio2017.csv" #DA CAMBIARE
dati = pd.read_csv(PATH)
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

print ("lunghezza", len(useless))

for i in range(0,len(useless)):
	del dati[useless[i]]


dati.pop('Unnamed: 109')
dati.to_csv("gennaio2017_pulito.csv")

#APPUNTI:
#unique carrier?
#città per nome o per ID?
