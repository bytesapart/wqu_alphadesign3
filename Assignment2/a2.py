# from __future__ import print_function
# from __future__ import absolute_import
import numpy as np
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import fix_yahoo_finance as yf

yf.pdr_override()
import datetime
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime as dt, timedelta
import time
import matplotlib.pyplot as plt
import math

# Anything from keras
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from tensorflow.python.keras import utils

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# to not display the warnings of tensorflow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set random seed for reproducibility
np.random.seed(123)

ticker = "AAPL"
start = dt.today() - timedelta(days=5 * 365)
end = dt.now()

df = web.DataReader(ticker, 'iex', start, end)

# convert index to datetime
df.index = pd.to_datetime(df.index)

# Convert close price to float and reshapre to column vector

# save stock close as float
stock_prices = df.close.values.astype('float32')

# reshape to column vector
stock_prices = stock_prices.reshape(len(stock_prices), 1)

train_size = int(len(stock_prices) * 0.7)
test_size = len(stock_prices) - train_size

scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices = scaler.fit_transform(stock_prices)

# save scaled close price for review/visualization
df['scaled_close'] = stock_prices

train, test = stock_prices[0:train_size, :], stock_prices[train_size:len(stock_prices), :]

print('Number of training samples/ test samples:', len(train), len(test))


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# convert Apple's stock price data into time series dataset
X_train, y_train = create_dataset(train, 7)
X_test, y_test = create_dataset(test, 7)

# Reshape the LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create and fit the lstm network
model = Sequential()
model.add(LSTM(4, input_shape=(7, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=3)

# Make Predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)


trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])

# Calculate Root Mean Squared Error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# get the test data from web dataset# get th
df_plot = df.iloc[len(df) - len(testPredict):len(df),:]

# setup date column
df_plot['date'] = df_plot.index
df_plot.reset_index(drop=True, inplace=True)

# add the predictions in
df_plot['prediction'] = pd.DataFrame(testPredict)
df_plot['date'] = pd.to_datetime(df_plot['date'])

# reset index
df_plot.set_index('date', inplace=True)

df_plot['close'].plot(figsize=(15,6), color="green")
df_plot['prediction'].plot(figsize=(15,6), color="red")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
plt.close()