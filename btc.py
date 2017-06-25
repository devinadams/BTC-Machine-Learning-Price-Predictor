# BTC Linear Regression price predictor
# Written By Devin Adams

from __future__ import print_function
import pandas as pd
import os
import time
import quandl, datetime, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

class BTC_LinearRegression():
    
    style.use('ggplot') # specify plot style

    def get_data(self):
        authToken = "YOUR QUANDL AUTH TOKEN"
        data = quandl.get("BCHARTS/COINBASEUSD", authtoken=authToken)

        data = data[['Open', 'Close', 'High', 'Low', 'Volume (BTC)']]

        data['HL_PCT'] = (data['High'] - data['Close']) / data['Close'] * 100.0
        data['PCT_change'] = (data['Close'] - data['Open']) / data['Close'] * 100.0

        data = data[['Close', 'HL_PCT', 'PCT_change', 'Volume (BTC)']]
        data.fillna(-99999, inplace=True)
        return data

    def start_regression(self):
        X = np.array(self.data.drop(['projected_price'], 1))
        X = preprocessing.scale(X)
        X_lately = X[-self.forecast_out:]
        X = X[:-self.forecast_out]
        self.data.dropna(inplace=True)
        y = np.array(self.data['projected_price'])
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.99)
        clf = LinearRegression(n_jobs=20) # train data
        clf.fit(X_train, y_train) # fit trained data?
        accuracy = clf.score(X_test, y_test)
        forecast_set = clf.predict(X_lately)
        return forecast_set, accuracy

    def project_future_time(self):
        self.data['Forecast'] = np.nan
        self.last_date = self.data.iloc[-1].name
        self.last_unix = time.mktime(self.last_date.timetuple())
        self.one_day = 86400
        self.next_unix = self.last_unix + self.one_day
        for i in self.forecast_set:
            self.next_date = datetime.datetime.fromtimestamp(self.next_unix)
            self.next_unix += 86400
            self.data.loc[self.next_date] = [np.nan for _ in range(len(self.data.columns)-1)]+[i]

    def plot_data(self):
        self.data['Close'].plot()
        self.data['Forecast'].plot()
        plt.plot(self.forecast_out)
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def __init__(self):     
        self.data = self.get_data()
        self.forecast_out = int(math.ceil(.01*len(self.data)))
        self.forecast_col = 'Close'
        self.data['projected_price'] = self.data[self.forecast_col].shift(-self.forecast_out)
        self.forecast_set, self.accuracy = self.start_regression()
        self.project_future_time()
        self.accuracy_check_var = .95
        if self.accuracy > self.accuracy_check_var:
            self.accuracy *= 100
            print("\n")
            print("Accuracy: ", self.accuracy, "%")
            print("\n")
            print("BTC Forecast Set: ", self.forecast_set)
            print("\n")
            print(self.forecast_out)
            print(self.data['Forecast'].tail())
            print("\n")
        else:
            pass
        #self.plot_data()

def main():
    test = int(input("Enter the number of times to train the model: "))
    while test > 0:
        try:
            btc = BTC_LinearRegression()
            test = test - 1
            if btc.accuracy > btc.accuracy_check_var:
                answer = raw_input("Do you wish to plot the data?: Y/N   ")
                answer = answer.upper()
                if answer == 'Y':
                    btc.plot_data()
                elif answer == 'N':
                    print("\n")
                    print("Accuracy:", btc.accuracy, "%")
                    print(btc.data['Forecast'].tail())
                    print("User skipped plotting.")
                    print("\n")
            else:
                pass
        except Exception as e:
            print("ERROR", e)
            pass
main()