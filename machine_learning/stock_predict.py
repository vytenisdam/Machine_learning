import yfinance as yahoofinance
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


style.use('ggplot')

df = yahoofinance.Ticker('AGNC').history(period='max')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
last_date = df.index[-1]
# Change in percent Close position and Highest position of the day.
df['High_close_percent'] = (df['High'] - df['Close']) / df['Close'] * 100.0
# Daily percent change.
df['Percent_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
# Some kind of Volatility through the day. Change between Lowest and Highest price.
df['Volatility'] = abs((df['High'] - df['Low']) / df['Low'] * 100.0)
df1 = df[['Close', 'High_close_percent', 'Percent_change', 'Volume', 'Volatility']]
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
# print(df[['Close','label']].tail(60))
# Removes NA entries


X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)# Drops two months because thers no data to put in labe
print(df[['Close','label']].tail(60))
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Support vector machine
# clf = svm.SVR(kernel='poly')
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print(accuracy)

#Linear regresion without cross validation
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
# print(accuracy)

forecast_set = clf.predict(X_lately)

# print(forecast_set, accuracy, forecast_out)

# df['Forecast'] = np.nan
# last_date = df.iloc[-1].name
# last_unix = last_date.timestamp()
# one_day = 86400
# next_unix = last_unix + one_day
# #
# for i in forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Forecast'] = np.nan
# print(df['Forecast'].tail(60))
# last_date = df.index[-1]  # Assuming the index is a datetime index
one_day = datetime.timedelta(days=1)
next_date = last_date + one_day

for i in forecast_set:
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    next_date += one_day
print(df[['Close','Forecast', 'label']].tail(60))
# Plotting the graph
df1['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# # Cross validation
# clf = LinearRegression()
# scores = cross_val_score(clf, X, y, cv=5)
# print(f'Cross validation scores of 5 fold {scores}, mean score is {sum(scores)/len(scores)}.')

# KFold from documentation, shows splits
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=2)
# for train, test in kf.split(X):
#     print("%s %s" % (train, test))
