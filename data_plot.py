import matplotlib.pyplot as plt
import pandas as pd
from data_preparation import clean_ds, useless1, useless2

PATH = "./DataSet/2015/On_Time_On_Time_Performance_2015_1/On_Time_On_Time_Performance_2015_1.csv"
df = pd.read_csv(PATH, delimiter="," ,encoding="utf-8-sig")
clean_ds(df,useless1)
df = df[(df['OriginState'] == 'NY') & (df['DestState'] == 'CA')]
clean_ds(df,useless2)
df["date_time"]=df["FlightDate"].map(str)+df["DepTime"].map(str)
df.pop("Unnamed: 109")
df.sort_values(['FlightDate','DepTime'],axis=0,ascending=[True, True],inplace = True)
df.to_csv("voliGennaio2015_NY_CA.csv",sep = ';',decimal = ',')

df.plot(x = 'date_time', y = 'ArrDelayMinutes', kind = 'line')
#plt.savefig("gennaio2015_NY_CA.png")
plt.show()


#dati per plot
