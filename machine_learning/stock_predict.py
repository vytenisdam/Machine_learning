import yfinance as yahoofinance
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

df = yahoofinance.Ticker('AVGO').history(period='max')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Change in percent Close position and Highest position of the day.
df['High_close_percent'] = (df['High'] - df['Close']) / df['Close'] * 100.0
# Daily percent change.
df['Percent_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
# Some kind of Volatility through the day. Change between Lowest and Highest price.
df['Volatility'] = abs((df['High'] - df['Low']) / df['Low'] * 100.0)

df = df[['Close', 'High_close_percent', 'Percent_change', 'Volume', 'Volatility']]

forecast_col = 'Close'

# Fills NaN values with "-99999", to not lose data, not to delete cols or rows with NaN datatype
df.fillna(-9999, inplace=True)

# Determines approximately 1% of the total number of rows in dataframe.
forecast_out = int(math.ceil(0.01*len(df)))

# Shifts the rows to 'label' col value would be 'Close' price of tomorrow
# or any other day by specifying forecast_out value 0.01. (0.01 - one percent of length to the future,
# 0.1 - ten percent length of the df to the future)
df['label'] = df[forecast_col].shift(-forecast_out)

# Removes NA entries
df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Support vector machine
clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#Linear regresion without cross validation
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Cross validation
clf = LinearRegression()
scores = cross_val_score(clf, X, y, cv=5)
print(f'Cross validation scores of 5 fold {scores}, mean score is {sum(scores)/len(scores)}.')
