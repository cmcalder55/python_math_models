# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:56:19 2022

@author: camer
"""
import datetime as dt
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression

def predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps):
    empty = [float('nan')]*len(timestamps)

    ts = knownTimestamps + timestamps
    hum = humidity + empty

    ts = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M') for t in ts]
    ts = [dt.datetime.strftime(t, '%m%d%H') for t in ts]

    allday = sorted(list(map(lambda i,j: (i,j), ts, hum)))

    df = pd.DataFrame(allday,columns=['timestamp','humidity']).astype({'humidity': float})

    missing = df[df['humidity'].isna()]

    df.fillna(method ='ffill', inplace = True)

    x = np.array(df['timestamp']).reshape(-1, 1)
    y = np.array(df['humidity']).reshape(-1, 1)

    regr = LinearRegression().fit(x,y)  

    return (regr.predict(missing['timestamp'].values.reshape(-1, 1)).tolist())

# Psychometric Testing 
def jobOffers(scores, lowerLimits, upperLimits):
    
    offers = [0]*len(lowerLimits)
    
    for i in range(len(lowerLimits)):
        for k in scores:
            if k in range(lowerLimits[i], upperLimits[i]+1):
                offers[i] += 1
    return offers


def predictTemperature(startDate, endDate, temperature, n):
    temp_len = len(temperature)
    p = int(temp_len/24)
    x = list(range(1,(24*p)+1))
    
    reg = LinearRegression()
    reg.fit(np.asarray(x).reshape(-1,1), temperature)
    
    a = x[-1] + 1
    b = list(range(a,a+temp_len))
    
    return(reg.predict(np.asarray(b).reshape(-1,1)).tolist())


if __name__ == "__main__":

# Psychometric testing data
scores = [4,8,7]
lowerLimits = [2,4]
upperLimits = [8,4]

# Missing humidity values testing data
startDate = "2013-01-01"
endDate = "2013-01-01"
knownTimestamps = ['2013-01-01 00:00','2013-01-01 01:00','2013-01-01 02:00','2013-01-01 03:00','2013-01-01 04:00',
                   '2013-01-01 05:00','2013-01-01 06:00','2013-01-01 08:00','2013-01-01 10:00','2013-01-01 11:00',
                   '2013-01-01 12:00','2013-01-01 13:00','2013-01-01 16:00','2013-01-01 17:00','2013-01-01 18:00',
                   '2013-01-01 19:00','2013-01-01 20:00','2013-01-01 21:00','2013-01-01 23:00']
humidity = ['0.62','0.64','0.62','0.63','0.63','0.64','0.63','0.64','0.48','0.46','0.45','0.44','0.46','0.47','0.48',
            '0.49','0.51','0.52','0.52']
timestamps = ['2013-01-01 07:00','2013-01-01 09:00','2013-01-01 14:00','2013-01-01 15:00','2013-01-01 22:00']
measHumidity = ['0.64','0.55','0.44','0.44','0.52']

# Predicting temperature testing data
temperature=[10.0,11.1,12.3,13.2,14.8,15.6,16.7,17.5,18.9,19.7,20.7,21.1,
             22.6,23.5,24.9,25.1,26.3,27.8,28.8,29.6,30.2,31.6,32.1,33.7]
startDate = '2013-01-01'
endDate = '2013-01-01'
n = 1

jobOffers(scores, lowerLimits, upperLimits)
predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps)
predictTemperature(startDate, endDate, temperature, n)
