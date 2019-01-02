# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:53:14 2018

@author: rmankopf
"""
import hmac
import numpy as np
import pandas as pd
import time as time
import hashlib
import requests
from datetime import date, timedelta
import time
import datetime
import json
import bitmex

api_key = **
api_secret = bytes(**, 'latin-1')



### Get the Orderbook
base_url = 'https://testnet.bitmex.com'
verb = 'GET'
function_url = '/api/v1/orderBook/L2?symbol=XBT&depth=50'
url = base_url + function_url
expires = int(round(time.time()) + 5)

message = bytes(verb + function_url + str(expires)  + '', "latin-1")
signature = hmac.new(api_secret, message, digestmod=hashlib.sha256).hexdigest()
headers = {'api-expires': str(expires),
               'api-key': api_key,
               'api-signature': signature}
r = requests.get(url, headers=headers)
data = r.json()
price_data = pd.DataFrame.from_dict(data)

### Get Trade Buckets for 1 minute
start_date = "&startTime=" + "2018-12-03%2019%3A00%3A00"
end_date = "&endTime=" + "2018-12-03%2020%3A00%3A00"
base_url = 'https://testnet.bitmex.com'
verb = 'GET'
#function_url = "/api/v1/trade/bucketed?binSize=1m&partial=true&symbol=XBT&count=100&reverse=true&startTime=2018-12-03%2019%3A00%3A00&endTime=2018-12-03%2020%3A00%3A00"
function_url = "/api/v1/trade/bucketed?binSize=1m&partial=true&symbol=XBT&count=100&reverse=true" + start_date + end_date
url = base_url + function_url 
expires = int(round(time.time()) + 5)

message = bytes(verb + function_url + str(expires)  + '', "latin-1")
signature = hmac.new(api_secret, message, digestmod=hashlib.sha256).hexdigest()
headers = {'api-expires': str(expires),
               'api-key': api_key,
               'api-signature': signature}
r = requests.get(url, headers=headers)
data = r.json()
price_data_1_minute = pd.DataFrame.from_dict(data)

### Get Trade Buckets for 5 minute
start_date = "&startTime=" + "2018-12-03%2017%3A00%3A00"
end_date = "&endTime=" + "2018-12-03%2020%3A00%3A00"
base_url = 'https://testnet.bitmex.com'
verb = 'GET'
#function_url = "/api/v1/trade/bucketed?binSize=1m&partial=true&symbol=XBT&count=100&reverse=true&startTime=2018-12-03%2019%3A00%3A00&endTime=2018-12-03%2020%3A00%3A00"
function_url = "/api/v1/trade/bucketed?binSize=5m&partial=true&symbol=XBT&count=100&reverse=true" + start_date + end_date
url = base_url + function_url 
expires = int(round(time.time()) + 5)

message = bytes(verb + function_url + str(expires)  + '', "latin-1")
signature = hmac.new(api_secret, message, digestmod=hashlib.sha256).hexdigest()
headers = {'api-expires': str(expires),
               'api-key': api_key,
               'api-signature': signature}
r = requests.get(url, headers=headers)
data = r.json()
price_data_5_minute = pd.DataFrame.from_dict(data)
price_data_5_minute.to_pickle("D:/Users/rmankopf/Desktop/Algorithmic Trading/Bitmex_Trading_Data.pkl")


### Plot Line Graphs
import matplotlib.pyplot as plt
plt.plot( "high", data= price_data_1_minute, marker='', markerfacecolor='blue', color='skyblue', linewidth=2)
plt.plot( "low", data=price_data_1_minute, marker='', color='olive', linewidth=2)
plt.legend()


### Plot Candlestick Charts
import plotly.plotly as py
import plotly.graph_objs as go

import pandas_datareader as web
from datetime import datetime

df = web.DataReader("aapl", 'morningstar').reset_index()

trace = go.Candlestick(x=df.Date,
                       open=df.Open,
                       high=df.High,
                       low=df.Low,
                       close=df.Close)
data = [trace]
py.iplot(data, filename='simple_candlestick')

###########################################################################################
#https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
###########################################################################################
                        ####Compute Functions####
###########################################################################################
def moving_average(df,param, n):
    """Calculate the moving average for the given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    MA = pd.Series(df[param].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(MA)
    return df

def exponential_moving_average(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EMA = pd.Series(df['close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    df = df.join(EMA)
    return df


def momentum(df, n):
    """
    
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['close'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df

def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
        DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))    
    df = df.join(RSI)
    return df

def bollinger_bands(df, n):

    MA = pd.Series(df['close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['close'].rolling(n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df

def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def row_delta(df):
    df_close_diff = pd.Series(df['close'] - df['close'].shift(1),  name="Delta_Close")
    df_open_diff = pd.Series(df['open'] - df['open'].shift(1),  name="Delta_Open")
    df_high_diff = pd.Series(df['high'] - df['high'].shift(1),  name="Delta_High")
    df_low_diff = pd.Series(df['low'] - df['low'].shift(1),  name="Delta_Low")
    df_turnover_diff = pd.Series(df['turnover'] - df['turnover'].shift(1),  name="Delta_Turnover")
    df_volume_diff = pd.Series(df['volume'] - df['volume'].shift(1),  name="Delta_Volume") 
    df = df.join(df_close_diff)
    df = df.join(df_open_diff)
    df = df.join(df_high_diff)
    df = df.join(df_low_diff)
    df = df.join(df_turnover_diff)
    df = df.join(df_volume_diff)
    return df

def RSI(df, length):
      # Get the difference in price from previous step
       delta = df['close'].diff()
       # Get rid of the first row, which is NaN since it did not have a previous
       # row to calculate the differences
       delta = delta[0:]
       # Make the positive gains (up) and negative gains (down) Series
       up, down = delta.copy(), delta.copy()
       up[up < 0.0] = 0.0
       down[down > 0.0] = 0.0
       # Calculate the EWMA
       roll_up1 = up.ewm(com=(length-1), min_periods=length).mean()
       roll_down1 = down.abs().ewm(com=(length-1), min_periods=length).mean()
       # Calculate the RSI based on EWMA
       RS1 = roll_up1 / roll_down1
       RSI1 = 100.0 - (100.0 / (1.0 + RS1))
       RSI = pd.Series(RSI1, name='RSI_TV' + str(length)) 
       #return(RSI)
       df = df.join(RSI)
       return (df)

###########################################################################################

###########################################################################################

   
# Check indicators and compute 

def check_markets(interval):
    today = str(datetime.datetime.now().date())
    if (interval == "1m"):
        call_back = 1
    elif (interval == "5m"):
        call_back = 3
    elif (interval == "1h"):
        call_back = 32
    elif (interval == "1d"):
        call_back = 750
    else:
        call_back = 1               
                            
    yesterday = str(datetime.datetime.now().date() - timedelta(call_back))
    end_hour = str(datetime.datetime.now().hour)
    start_hour = str(datetime.datetime.now().hour)
    end_minute = str(datetime.datetime.now().minute)
    start_minute = str(datetime.datetime.now().minute)
    start_date = "&startTime=" + yesterday + "%20" + start_hour  + "%3A" + start_minute +"%3A00"
    end_date = "&endTime=" + today +"%20"+ end_hour + "%3A"+ start_minute + "%3A00"
    base_url = 'https://testnet.bitmex.com'
    verb = 'GET'               
    function_url = "/api/v1/trade/bucketed?binSize="+ interval + "&partial=true&symbol=XBT&count=750&reverse=true" + start_date + end_date
    url = base_url + function_url 
    expires = int(round(time.time()) + 5)
    
    message = bytes(verb + function_url + str(expires)  + '', "latin-1")
    signature = hmac.new(api_secret, message, digestmod=hashlib.sha256).hexdigest()
    headers = {'api-expires': str(expires),
                      'api-key': api_key,
                      'api-signature': signature}
    r = requests.get(url, headers=headers)
    data = r.json()
    price_data_n_minute = pd.DataFrame.from_dict(data)
    price_data_n_minute = price_data_n_minute.sort_values(["timestamp"]).reset_index()
    price_data_n_minute["index"] = price_data_n_minute.index
    return(price_data_n_minute)

def calculate_indices(data):
    data = RSI(data,14)      
    data = macd(data, 12, 26)
    data["Price_OCD"] = (data["open"] + data["close"]) /2
    data = moving_average(data, "Price_OCD", 50 )
    data = moving_average(data, "Price_OCD", 150 )
    data = moving_average(data, "Price_OCD", 250 )
    data = momentum(data,14)
    data = row_delta(data)
    data = bollinger_bands(data, 20)
    data = relative_strength_index(data, 14)
    return(data)


def compute_signal(data):
    trade_relevance_0 = data.iloc[len(data)-1]
    trade_relevance_1 = data.iloc[len(data)-2]
    trade_relevance_2 = data.iloc[len(data)-3]
    trade_relevance_4 = data.iloc[len(data)-4]
    
    #Simple Strategy
    if (
        trade_relevance_0["RSI_TV14"] < 0.4 and 
        np.round(trade_relevance_0["MACD_12_26"]) > np.round(trade_relevance_0["MACDsign_12_26"])
        ):
            signal_0 = 1

    if (
        trade_relevance_0["RSI_TV14"] > 0.7 and 
        np.round(trade_relevance_0["MACD_12_26"]) < np.round(trade_relevance_0["MACDsign_12_26"])
        ):
            signal_0 = 0
        
    if (
        trade_relevance_1["RSI_TV14"] < 0.4 and 
        np.round(trade_relevance_1["MACD_12_26"]) > np.round(trade_relevance_1["MACDsign_12_26"])
        ):
            signal_1 = 1

    if (
        trade_relevance_1["RSI_TV14"] > 0.7 and 
        np.round(trade_relevance_1["MACD_12_26"]) < np.round(trade_relevance_1["MACDsign_12_26"])
        ):
            signal_1 = 0
        
    position = signal_0 - signal_1
    return(position)
    #if position = 1 -> buy if position = -1 -> sell
    
def trade_long(price_level):
    client = bitmex.bitmex(test= True, api_key= **, api_secret= **)
    client.Order.Order_new(symbol='XBTUSD', orderQty=100, price= price_level).result()
    
def trade_short(price_level):
    client = bitmex.bitmex(test= True, api_key= **, api_secret= **)
    client.Order.Order_new(symbol='XBTUSD', orderQty= -100, price= price_level).result()



###Inspiration#######

#https://medium.com/@ilia.krotov/margin-trading-robot-on-the-bitmex-exchange-d94dd46f82c5
