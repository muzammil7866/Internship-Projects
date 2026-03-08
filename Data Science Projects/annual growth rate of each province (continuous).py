import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("sub-division_population_of_pakistan.csv")
print(df.head())
df.info()

df['ANNUAL GROWTH RATE'] = df['ANNUAL GROWTH RATE (URBAN)'] + df['ANNUAL GROWTH RATE (RURAL)']
df = df[['PROVINCE', 'ANNUAL GROWTH RATE']]
print(df)

provinces = df['PROVINCE'].unique()
provinces = provinces[0:4]

data = {}

for i in provinces:

    AGR = 0
    count = 0

    for j in df.index:
        if(df.loc[j]['PROVINCE'] == i):
            AGR += df.loc[j]['ANNUAL GROWTH RATE']
            count += 1

    data[i] = AGR/count

print(data)

finalDataFrame = pd.Series(data)
print(finalDataFrame)

finalDataFrame.plot(kind = 'bar')

plt.show()


