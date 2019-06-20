#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:49:19 2018

@author: IanFan
"""

import fix_yahoo_finance as yf
from weights import Balance as BA
import pandas as pd

from sharpe import Screen
#import pandas as pd
#from pandas.tseries.offsets import CustomBusinessMonthBegin

Ticker=pd.read_csv('Ticker.csv')

REIT=Ticker.Innovation.values.tolist()

REIT.append('FFTY')

data = yf.download(REIT,start='2007-01-01',end='2019-05-17')

#data = yf.download(['SPY'],start='2005-01-01',end='2019-04-22')

In=Ticker.Innovation.values.tolist()

In.append('FFTY')

data = yf.download(In,start='2007-01-01',end='2019-05-23')



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

#adjclose.drop([])
adjclose1=adjclose.dropna()

P=Screen(adjclose1,In)

Result=P.sharpe_ratio('bi','FFTY')

L=Screen.pick_sharpe(Result,10)

#Sharpe  Correlation
#AWK  0.279683     0.858460
#WEC  0.229105     0.913436
#XEL  0.203862     0.907971
#AEE  0.187987     0.904151
#CMS  0.163887     0.924004

#Sharpe  Correlation
#MKC  0.515382     0.592506
#CHD  0.467034     0.442862
#HSY  0.292108     0.631488
#SJM  0.284386     0.445731
#PG   0.282926     0.697304

#Sharpe  Correlation
#SUI  0.260754     0.706299
#ELS  0.207881     0.733267
#EXR  0.175193     0.722196
#DLR  0.129272     0.617164
#PLD  0.118385     0.813388



selected=['VDC','VPU','SCHH','VHT','SPY','SPXU','QQQ']

data = yf.download(selected,start='2007-01-01',end='2019-05-17')

adjclose=data['Adj Close']

adjclose1=adjclose.dropna()


def main():
    
    selected=['VDC','VPU','SCHH','VHT','SPXU','QQQ']
    target=0.002
    capital=25000
    bench='SPY'
    
#time option has bi or None, bi means bi weekly rebalance the data, None means monthy rebalance
    timoption='bi'
    
#option1 has the chocie of None, 1, 2 which refers to min variance, max sharpo ratio, max return given covariance  
    option1=2
    
    BA.back_testing(adjclose1,selected,capital,target,option1,bench,timoption)

    weights=BA.find_weights(adjclose1,selected,target,timoption)
    table=BA.find_shares(adjclose1, selected, weights,capital)
    print(table)

#mdata=Adjdata.resample('M').mean()

main()
#the red line the if we predicted vol 100% correct, and blue dot line is the prediction line
#the blue dash line is sp100





'Save the data into local database'


'Calculate the weights on each asset'


'Save weights and predicted return into local database'


'Calculate the transaction fees'


