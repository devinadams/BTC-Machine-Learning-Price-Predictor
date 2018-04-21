#The MIT License (MIT)

# Copyright (c) 2018 Devin Adams <github.com/devinatoms/>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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

dataList = np.array([])


class BTC_LinearRegression():

    def get_data(self):
        authToken = "YOUR AUTH TOKEN HERE"
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

    def __init__(self):
        global dataList     
        self.data = self.get_data()
        self.forecast_out = int(math.ceil(.0023*len(self.data))) # This is how far back in the data it looks
        self.forecast_col = 'Close'
        self.data['projected_price'] = self.data[self.forecast_col].shift(-self.forecast_out)
        self.forecast_set, self.accuracy = self.start_regression()
        self.project_future_time()
        self.newList = dataList
        self.accuracy_check_var = .98 # set accuracy threshold here
        if self.accuracy > self.accuracy_check_var:
            self.accuracy *= 100
            self.smallest_forecast_value_index = self.forecast_set.argmin()
            self.average_forecast_value = self.forecast_set.mean()
            dataList = np.append(dataList, self.forecast_set)
            self.newList = dataList.tolist()
            average = sum(self.newList) / float(len(self.newList))
            print("Accuracy: ", self.accuracy, "%")
            print("Mean of set: ", average)
            print('\n')
            print("Set: ", dataList)
            print("\n")
        else:
            pass

def main():
    global dataList
    try:
        test = int(input("Enter the number of times you would like to calculate: "))
    except Exception as e:
        print("Please enter a number this time. ")
        main()
        pass
    while test > 0:
        try:
            btc = BTC_LinearRegression()
            if btc.accuracy > btc.accuracy_check_var:
                test = test - 1
                max_val = max(btc.newList)
                min_val = min(btc.newList)
                print(btc.newList)
                print("\n")        
                print("Average of Prediction: $", sum(btc.newList) / float(len(btc.newList)))
                print("Highest Prediction: $", max_val)
                print("Lowest Prediction: $", min_val)
                print("\n")
        except Exception as e:
            print("ERROR", e)
            pass
main()
