#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:49:19 2018

@author: IanFan
"""

'API import the new month adjusted close price data from Yahoo Finance'
import fix_yahoo_finance as yf
from weights import Balance as BA
#import pandas as pd
#from pandas.tseries.offsets import CustomBusinessMonthBegin
data = yf.download(['SPY','MCHI','QQQ','ICLN','QCLN','IBB','XBI','FBT','IVV','IYR','SCHH'],start='2007-01-01',end='2019-04-22')

data = yf.download(['SPY'],start='2005-01-01',end='2019-04-22')


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
adjclose1=adjclose.dropna()

def main():
    selected=['SPY','QQQ','IBB','XBI','FBT','IYR','SCHH']
    target=0.005
    capital=50000
    bench='SPY'
    
#time option has bi or None, bi means bi weekly rebalance the data, None means monthy rebalance
    timoption=None
    
#option1 has the chocie of None, 1, 2 which refers to min variance, max sharpo ratio, max return given covariance  
    option1=2
    
    BA.back_testing(adjclose1,selected,capital,target,option1,bench,timoption)
    weights=BA.find_weights(adjclose1,selected,target,timoption)
    table=BA.find_shares(adjclose1, selected, weights,capital)

#mdata=Adjdata.resample('M').mean()

main()
#the red line the if we predicted vol 100% correct, and blue dot line is the prediction line
#the blue dash line is sp100
table




'Save the data into local database'


'Calculate the weights on each asset'


'Save weights and predicted return into local database'


'Calculate the transaction fees'


