#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
#from tensorflow.python.compiler import tensorrt as trt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
import pandas
import os
import datetime as dt
import yfinance
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
import pandas_market_calendars as mcal

# Commodity Channel Index Python Code
# Load the necessary libraries
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance
import pandas as pd

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[ ]:





# In[ ]:


#start_ = datetime.datetime.strptime(str(dates.iloc[0].values[0]), '%Y%m%d').strftime('%Y-%m-%d')
#end_ = datetime.datetime.strptime(str(dates.iloc[-1].values[0]), '%Y%m%d').strftime('%Y-%m-%d')
start_ = "2009-01-02"
end_ = "2020-08-14"


# In[ ]:


#data.tic.unique()
symbols = ['PFE', 'DIS', 'MRK', 'VZ', 'GS', 'IBM', 'AXP', 'CSCO', 'CAT',        'CVX', 'RTX', 'DD', 'INTC', 'MSFT', 'NKE', 'WMT', 'JNJ', 'BA',        'AAPL', 'MCD', 'WBA', 'TRV', 'HD', 'PG', 'XOM', 'MMM', 'KO', 'JPM',        'V', 'UNH']


# In[ ]:


#20151001)&(data.datadate <= 20200707
validation_start = "2015-10-01"
trading_start = datetime.datetime.strptime("2016/01/01", '%Y/%m/%d').strftime('%Y-%m-%d')
end_trade_date = "2020-07-07"
#data_.index


# In[ ]:





# In[ ]:



n0 = 9
n1 = 12
n2 = 26
n3 = 14


#dates = pd.DataFrame(data.datadate.unique())


# Create a calendar
nyse = mcal.get_calendar('NYSE')



print(start_)
print(validation_start)
print(trading_start)
print(end_trade_date)
print(end_)


#stocks = data.tic.unique()
#symbols=stocks
start = start_#date_time[0]
early_start = (datetime.datetime.strptime(start, '%Y-%m-%d') - pd.tseries.offsets.BusinessDay(n = np.max([n0,n1,n2,n3])+5)).strftime('%Y-%m-%d')
end = (datetime.datetime.strptime(str(end_), '%Y-%m-%d') + pd.tseries.offsets.BusinessDay(n = 1)).strftime('%Y-%m-%d')#date_time[-1]

early = nyse.schedule(start_date=early_start, end_date=end)


# In[ ]:


dates = early.index


# In[ ]:





# In[ ]:


#dates_ = []

#for i in range(0,len(dates)):
    #print(i)
    #dates['date'].values[i] = str(dates['date'].values[i].copy())
    #dates_.append(str(dates['date'].values[i].copy()))
#list((dates['date']).values)[0]


# In[ ]:


date_time = []
for i in range(0,len(dates)):
    date_time.append(dates[i].strftime('%Y-%m-%d'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


n_forward = 40
start_date = start
end_date = end

dataSet = pd.DataFrame()

#testSize = 3
testSize = len(symbols)

#for i in symbols:
for i in pd.DataFrame(symbols).sample(testSize).values.ravel():
    print(i)
    
    name = i
    ticker = yfinance.Ticker(i)
    data_ = ticker.history(interval="1d",start=early_start,end=end_date)
    data_['tic'] = i
    
    '''
    exp1 = data_.Close.ewm(span=n1, adjust=False).mean()
    exp2 = data_.Close.ewm(span=n2, adjust=False).mean()
    macd = (exp1-exp2)/exp2
    exp3 = macd.ewm(span=n0, adjust=False).mean()
    macd_ = pd.concat([macd,exp3],axis=1)
    macd_.columns =["macd","signal"]
    data_[['ADX']] = ADX(data_,n3)[['ADX']]
    rsi_ = rsi(data_[['Close']])
    rsi_.columns = ['rsi']
    rsi_    #print(temp)
    cci_ = CCI(data_.copy(),n3)[['TP','sma','mad','CCI']]
    temp1 = data_[['tic','Close']]
    temp1[['datadate']] = temp1.index
    temp1.columns = ['tic','adjcp','datadate']
    '''
    
    '''
    df = data_.reset_index()
    df.columns = ['datadate', 'Open', 'High', 'Low', 'adjcp', 'Volume', 'Dividends','Stock Splits', 'tic', 'ADX']
    #df.index = df['datadate']
    
    #plt.plot(temp, macd, label='AMD MACD', color = '#EBD2BE')
    #plt.plot(df.ds, exp3, label='Signal Line', color='#E5A4CB')
    #plt.legend(loc='upper left')
    #plt.show()    
    
    turb_ = add_turbulence(df)[['turbulence']]
    turb_.index = df['datadate']
    #print(turb_.tail())
    '''
    
    '''
    #data_new = pd.concat([data_,cci_,macd_,rsi_,turb_],axis=1).reset_index()
    data_new = pd.concat([data_,cci_,macd_,rsi_],axis=1).reset_index()
    #data_new.columns = ['datadate','Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'tic', 'ADX', 'TP', 'sma', 'mad', 'CCI', 'macd', 'signal', 'rsi', 'turbulence']
    data_new.columns = ['datadate','Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'tic', 'ADX', 'TP', 'sma', 'mad', 'CCI', 'macd', 'signal', 'rsi']
    '''
    #dataSet = pd.concat([dataSet,data_new],axis=0,ignore_index=True)
    dataSet = pd.concat([dataSet,data_.reset_index()],axis=0,ignore_index=True)
    


# In[44]:


dataSet[['ajexdi']]= 1 


# In[45]:


dataSet[['datadate']] = dataSet.set_index('Date').index.strftime('%Y%m%d')


# In[ ]:


#dataSet[['datadate']].set_index('datadate').index.strftime('%Y-%m-%d')


# In[46]:


dataSet.drop('Date',inplace=True,axis=1)
#dataSet


# In[47]:


dataSet.columns = ['prcod','prchd','prcld','prccd','cshtrd','Dividends','Stock Splits','tic','ajexdi','datadate']


# In[48]:


TRAINING_DATA_FILE = r"C:\Users\User\Documents\wiki\wiki\dev\python\python-ml\code\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020\data\dow_30_2009_2020.csv"

dataSet.to_csv(TRAINING_DATA_FILE, index=True)


# In[ ]:


#[datadate,tic,adjcp,open,high,low,volume,macd,rsi,cci,adx,turbulence]

'''
truncatedDataSet = pd.DataFrame()

for i in dataSet.tic.unique():
    print(i)
    df = dataSet[dataSet.tic==i]
    df2 = df[df.set_index('datadate').index.get_loc(datetime.datetime.strptime(start_, "%Y-%m-%d").date().strftime('%Y%m%d')):df.set_index('datadate').index.get_loc(datetime.datetime.strptime(end_, "%Y-%m-%d").date().strftime('%Y%m%d'))]
    #print(df2.head(30))
    #print(df2.tail())
    truncatedDataSet = pd.concat([truncatedDataSet.copy(),df2],axis=0)
    
#'low', 'adjcp', 'open', 'volume', 'cci', 'adx', 'high'
truncatedDataSet.columns = ['datadate', 'open', 'high', 'low', 'adjcp', 'volume', 'Dividends', \
       'Stock Splits', 'tic', 'adx', 'TP', 'sma', 'mad', 'cci', 'macd', \
       'signal', 'rsi', 'turbulence']

truncatedDataSet[truncatedDataSet.columns].to_csv('done_data2.csv', index=True)
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




