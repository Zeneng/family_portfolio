#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:49:19 2018

@author: IanFan
"""

'API import the new month adjusted close price data from Yahoo Finance'
import fix_yahoo_finance as yf
#import pandas as pd
#from pandas.tseries.offsets import CustomBusinessMonthBegin
data = yf.download(['SPY','LQD','MCHI','QQQ','REM','ICLN','QCLN','IBB','XBI','FBT','IVV','FINX','XBI'],start='2016-01-01',end='2019-04-22')


'Convert the daily data into monthly data'
#month_index =data.index.to_period('M')
#
#min_day_in_month_index = pd.to_datetime(data.set_index(month_index, append=True).reset_index(level=0).groupby(level=0)['Open'].min())
#
#custom_month_starts = CustomBusinessMonthBegin(calendar = min_day_in_month_index)
#
#ohlc_dict = {'Open':'first','High':'max','Low':'min','Close': 'last','Volume': 'sum','Adj Close': 'last'}
#
#mthly_ohlcva = data.resample(custom_month_starts, how=ohlc_dict)
adjclose=data['Adj Close']

#mdata=Adjdata.resample('M').mean()





'Save the data into local database'


'Calculate the weights on each asset'


'Save weights and predicted return into local database'


'Calculate the transaction fees'


