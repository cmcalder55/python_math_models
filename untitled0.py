# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:46:53 2022

@author: camer
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier


def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        t=.4
        preds[preds > t] = 1
        preds[preds<= t] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)

if __name__ == '__main__':
    df = pd.read_csv("../cpe608/TSLA.csv",parse_dates=['Date']).set_index('Date')
    
    # df.plot(y='Close',use_index=True)
    # record actual stock prices
    data = df[['Close']]
    data = data.rename(columns = {'Close':'Actual_Close'})
    # use rolling method to compare each row to the previous row and assign 1 if greater or 0
    data["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    # shift forward one to predict tomorows prices using todays values
    tsla_prev = df.copy().shift(1)
    
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data = data.join(tsla_prev[predictors]).iloc[1:]
    
    # Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
    # model = SGDRegressor()
    model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
    
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    
    model.fit(train[predictors], train["Target"])
        
    predictions = backtest(data, model, predictors)


















