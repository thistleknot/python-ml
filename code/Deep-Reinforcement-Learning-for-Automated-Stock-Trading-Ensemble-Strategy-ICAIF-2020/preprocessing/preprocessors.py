import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def ma(Data, lookback, what, where):
    
    Data = Data.reset_index()
    #newData = Data.reset_index()
    for i in range(len(Data)):
        try:
            Data.loc[i, where] = (Data.loc[i - lookback + 1:i + 1, what].mean())

        except IndexError:
            pass
    #newData.index = Data.index
    return Data

def ema(Data, alpha, lookback, what, where):
    
    Data = Data.reset_index()
    # alpha is the smoothing factor
    # window is the lookback period
    # what is the column that needs to have its average calculated
    # where is where to put the exponential moving average
    
    alpha = alpha / (lookback + 1.0)
    beta  = 1 - alpha
    
    # First value is a simple SMA
    Data = ma(Data, lookback, what, where)
    
    # Calculating first EMA
    Data.loc[lookback + 1, where] = (Data.loc[lookback + 1, what] * alpha) + (Data.loc[lookback, where] * beta)    
    # Calculating the rest of EMA
    for i in range(lookback + 2, len(Data)):
            try:
                Data.loc[i, where] = (Data.loc[i, what] * alpha) + (Data.loc[i - 1, where] * beta)
        
            except IndexError:
                pass
    return Data

def volatility(Data, lookback, what, where):
    
    Data = Data.reset_index()
    for i in range(len(Data)):
        try:
            Data.loc[i, where] = (Data.loc[i - lookback + 1:i + 1, what].std())
    
        except IndexError:
            pass
        
    return Data

def augmented_BollingerBands(Data, boll_lookback, standard_distance, high, low):
    
    hi = ema(Data, 2, boll_lookback, high, 'high-ema')['high-ema']
  
    lo = ema(Data, 2, boll_lookback, low, 'low-ema')['low-ema']
  
    hv = volatility(Data, boll_lookback, high, 'high-vol')['high-vol']
    lv = volatility(Data, boll_lookback, low, 'low-vol')['low-vol']
   
    high_band = (hi+(standard_distance * hv))
    low_band = (lo+(standard_distance * lv))
    temp = pd.concat([pd.DataFrame(high_band),pd.DataFrame(low_band)],axis=1)
    temp.columns = ['high_band','low_band']
    temp.index = Data.index
   
    return temp

def wwma(values, n):
    """
     J. Welles Wilder's EMA 
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def atr(df, lookback, HIGH, LOW, CLOSE):
    data = df.copy()
    high = data[HIGH]
    low = data[LOW]
    close = data[CLOSE]
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, lookback)
    return atr

def calculate_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd','high_abband','low_abband']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume','high_abband','low_abband']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())
    #print(stock)
    
    #print(len(df))

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macdO = pd.DataFrame()
    macdS = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()
    
    atr = pd.DataFrame()
    bbands = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        
        temp_macdS = stock[stock.tic == unique_ticker[i]]['macds']
        temp_macdS = pd.DataFrame(temp_macdS)
        macdS = macdS.append(temp_macdS, ignore_index=True)        
        #print(macdS)
		
        #MACD = (12 Day EMA - 26 Day EMA) / 26 Day EMA
        temp_macdO = stock[stock.tic == unique_ticker[i]]['macd']		
        temp_macdO = pd.DataFrame(temp_macdO)
        macdO = macdO.append(temp_macdO, ignore_index=True)        
        #print(macdO)
        #(stock[stock.tic == unique_ticker[i]]['MACD_EMA_SHORT'] - stock[stock.tic == unique_ticker[i]]['MACD_EMA_LONG']) / stock[stock.tic == unique_ticker[i]]['MACD_EMA_LONG']
                
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)
        
        #print(stock[stock.tic == unique_ticker[i]].columns)
        #'''
        #temp_atr = stock[stock.tic == unique_ticker[i]]['atr']
        #print(stock.columns)
        temp_atr = stock[stock.tic == unique_ticker[i]]['atr']
        temp_atr = pd.DataFrame(temp_atr)
        atr = atr.append(temp_atr, ignore_index=True)
        
        #temp_bbands = augmented_BollingerBands(stock[stock.tic == unique_ticker[i]],20,2,'high','low')
        #temp_bbands = pd.DataFrame(temp_bbands)
        #temp_bbands.columns = ['high_bband','low_bband']
        #bbands = temp_bbands.append(temp_bbands, ignore_index=True)
		
    #print(macdS)
    df['macd'] = macdS
    df['macdO'] = macdO
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx
    df['atr'] = atr
    #df['high_bband'] = bbands['high_bband']
    #df['low_bband'] = bbands['low_bband']
    #df = pd.concat([df.copy(),bbands],axis=1)
    #print(df.tail())

    return df

def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # get data after 2009
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calculate_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calculate_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calculate_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










