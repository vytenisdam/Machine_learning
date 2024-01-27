import yfinance as yahoofinance
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split


# data = yahooFinance.Ticker('AVGO')
# df = data.history(period='max')
df = yahoofinance.Ticker('AVGO').history(period='max')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Getting two new columns of percent change and if the stock has increased!!!!CHECK what does this means
df['Higher_lower_percent'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['Percent_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'Higher_lower_percent', 'Percent_change', 'Volume']]

forecast_col = 'Close'
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])
# CROSS VALIDATION HOW TO MAKE IT WORK
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)