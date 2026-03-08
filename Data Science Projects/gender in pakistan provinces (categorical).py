import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("sub-division_population_of_pakistan.csv") #reads file
#print(df.info())
#print(df.columns)



df['TOTAL MALE'] = df['MALE (URBAN)'] +  df['MALE (RURAL)'] #total male for that province district
df['TOTAL FEMALE'] = df['FEMALE (URBAN)'] +  df['MALE (RURAL)']  #total female for that province district
df = df[['PROVINCE', 'TOTAL MALE', 'TOTAL FEMALE']] #updated and relevant dataframe

provinces = df['PROVINCE'].unique()
provinces = provinces[0:4] #gets the four required provinces
#print(provinces)
data = {} #the final dataframe to visualise
#calculating sum of male and female for each province as {data}
for i in provinces: #iterate over the required provinces

    #Iterates provinces
    data[i] = {} #each province has a dictionary containing info for genders: male or female

    sumMale = 0
    sumFemale = 0

    #calculating total male and female sum for that province
    for j in df.index: #iterate over all the indices
        if(df.loc[j]['PROVINCE'] == i): #check if that row is of the matching province
            sumMale += df.loc[j]['TOTAL MALE'] #add into the total male sum
            sumFemale += df.loc[j]['TOTAL FEMALE'] #add into the total female sum

    data[i]['MALE'] = sumMale
    data[i]['FEMALE'] = sumFemale

#print(data)

finalDataFrame = pd.DataFrame(data)
#print(finalDataFrame)

finalDataFrame.plot(kind = 'bar')
plt.show()

