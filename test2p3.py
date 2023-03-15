# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import scale

def plotPrice(df,n):
    
    plt.figure()
    
    title = 'Tesla Stock Price for the Past '+str(n)+' Days'
    df.plot(x='Date', y = 'Stock Price', title=title, legend=False)
    
    plt.xticks(rotation=70)
    plt.grid(True)
    
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.show()
    
def formatData(filepath,n):
    df = pd.read_csv(filepath, header=0)
    
    data = {'Date':df['Date'], 'Stock Price':df['Close']}
    df = pd.DataFrame(data)
    
    plotPrice(df,n)
    dates = pd.to_datetime(df['Date'])

    df['Date'] = (dates-dates.min())/np.timedelta64(1,'D')
    
    return  df, dates

def SGDregression(df):
    
    x = df['Date']
    y = df['Stock Price']
    
    X = np.array(x).reshape(-1,1)
    
    xtrain, xtest, ytrain, ytest=train_test_split(X, y, random_state=0, train_size = .75)
    # print(xtest)
    ind = np.concatenate((xtest,xtrain),axis=0)

    # X = np.array(range(30,91)).reshape(-1,1)
    X = scale(X)
    y = scale(y)
    
    xtrain, xtest, ytrain, ytest=train_test_split(X, y, random_state=0, train_size = .75)
    
    sgdr = SGDRegressor()
    sgdr.fit(xtrain, ytrain)
    
    ypred = sgdr.predict(xtest)
    # print(xtest)
    # ypred = sgdr.predict(np.array(range(30,91)).reshape(-1,1))
    
    v = np.concatenate((ypred,ytrain),axis = 0)
    
    dd = pd.DataFrame({'Day':pd.Series(ind.flatten()),
                        'Predicted Price':pd.Series(v.flatten()),
                        'Actual Price': y})
    
    dd = dd.set_index('Day').sort_index().reset_index()
    
    X_new = np.array(range(30,91)).reshape(-1,1)
    y_new = sgdr.predict(scale(X_new))
    
    
    return y, dd,y_new
    
def plotPredictions(df,dates):
    
    actual,dd,y_new = SGDregression(df)
    
    plt.plot(dates,actual,label="Actual")
    plt.plot(dates,dd['Predicted Price'],label="Predicted")
    
    plt.title("Predicted and Actual Stock Price, 30 Days")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    
    plt.xticks(rotation=70)
    
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()
    return dd
    
if __name__ == '__main__':
    
    # Predictions for 1 month
    data = "TSLA.csv"
    filepath = "../cpe608/" + data

    data, dates= formatData(filepath,30)
    
    dd=plotPredictions(data,dates)
    
    data['Date'] = dates
    df = data.set_index('Date')
    
    forecast = 90
    
    x=dd['Day']
    y=dd['Predicted Price']
    
    a,b = np.polyfit(x,y,1)
    
    plt.scatter(x,y)
    
    plt.plot(x,a*x+b)
    plt.title("Predicted Stock Price Line of Best Fit")
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    
    
    
    
    
    
    
    
    
    