# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:29:54 2023

@author: camer
"""
import numpy as np

class LinearRegression():
    def __init__(self):
        self.learning_rate = 1e-10
        self.epochs = 10000
        
    def yhat(self, X, w):
        return np.dot(w.T, X)
    
    def loss(self, yhat, y):
        L = 1/self.m * np.sum(np.power(yhat - y, 2))
        return L
        
    def gradient_descent(self, w, X, y, yhat):
        dldw = 2/self.m * np.dot(X, (yhat-y).T)
        w = w - self.learning_rate*dldw
        return w
        
    def main(self,X,y):
        x1 = np.ones((1, X.shape[1]))
        X = np.append(X, x1, axis=0)
        
        self.m = X.shape[1]
        self.n = X.shape[0]
        
        w = np.zeros((self.n, 1))
        
        for epoch in range(self.epochs+1):
            yhat = self.yhat(X,w)
            loss = self.loss(yhat, y)
            
            if epoch % 2000 == 0:
                print(f'cost at epoch {epoch} is {loss:.8}')
                
            w = self.gradient_descent(w, X, y, yhat)
            
        return w
    
    
if __name__ == '__main__':
    X = np.random.rand(1,500)
    y = 3*X + np.random.randn(1,500)*0.1
    reg = LinearRegression()
    w = reg.main(X,y)
    