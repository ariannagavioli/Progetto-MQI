import matplotlib.pyplot as plt
import pandas as pd

PATH = "/home/luca/Scrivania/PROGETTO/perfetto_plot.csv"
df = pd.read_csv(PATH, delimiter=";" ,encoding="utf-8-sig")
df.plot(x = 'date_time', y = 'ArrDelayMinutes', kind = 'line')
plt.savefig("gennaio2015_NY_CA.png")
plt.show()