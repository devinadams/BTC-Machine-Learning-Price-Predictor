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

class BTC_LinearRegression():
    def get_data(self):

        token = "YOUR QUANDL AUTH TOKEN GOES HERE"

        data = quandl.get("BCHAIN/MKPRU", authtoken = token )
        data = data[['Value']]
        data.fillna(-99999, inplace=True)
        return data

    def start_regression(self):
        X = np.array(self.data.drop(['projected_price'], 1))
        X = preprocessing.scale(X)
        X_lately = X[-self.forecast_out:]
        X = X[:-self.forecast_out]
        self.data.dropna(inplace=True)
        y = np.array(self.data['projected_price'])
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.9)
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
        self.data['Value'].plot()
        self.data['Forecast'].plot()
        plt.plot(self.forecast_out)
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
        
        
    def __init__(self):
        self.data = self.get_data()
        self.forecast_out = int(math.ceil(.0007*len(self.data))) # Usually use .0001, The higher the value the farther it regresses
        self.forecast_col = 'Value'
       # print(self.data[self.forecast_col].tail())
        self.data['projected_price'] = self.data[self.forecast_col].shift(-self.forecast_out)
        self.accuracy_check_var = .986 # Almost never returns over 98.5% accuracy (THIS CHANGES ALOT)
        self.forecast_set, self.accuracy = self.start_regression()
        self.project_future_time()
        self.storage = []
        self.holder = []

def main():
    data =[]
    try:
        counter = int(input("Enter number of times to try predicting data: "))
    except Exception as e:
        print("Please enter a numbers only this time: ")
        main()
    print("\n")
    print("Calculating please wait...")
    print("\n")
    while counter > 0:
        btc = BTC_LinearRegression()
        counter = counter - 1
        try:
            if btc.accuracy < btc.accuracy_check_var:
                print("Accuracy low, passing result")
                counter = counter + 1
                pass
            if btc.accuracy > btc.accuracy_check_var:
                data = data + btc.forecast_set.tolist()
                print(test, btc.forecast_set, "Accuracy:" , btc.accuracy, "%")
        except Exception as e:
            print(e)
        if counter == 0: 
            max_val = max(data)
            min_val = min(data)
            print(data)
            print("\n")        
            print("Average of Predictions: $", sum(data) / float(len(data)))
            print("Highest Prediction: $", max_val)
            print("Lowest Prediction: $", min_val)
            print("\n")
main()
