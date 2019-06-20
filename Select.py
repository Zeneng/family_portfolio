#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:26:37 2019

@author: IanFan
"""
import pandas as pd

import numpy as np

from importlib import reload

import fix_yahoo_finance as yf

import sharpe


Ticker=pd.read_csv('Ticker.csv')

Index_dict={'FINX': 'Fintech',
 'VAW': 'Material',
 'VCR': 'Discretion',
 'VDC': 'Consumer',
 'VDE': 'Energy',
 'VFH': 'Finance',
 'VGT': 'Information',
 'VHT': 'Health',
 'VIS': 'Industry',
 'VNQ': 'Reit',
 'VOX': 'Communication',
 'VPU': 'Utility',
 'SPY':'Market'}



class top_down(object):
    
    def __init__(self, index_dict, start_time, end_time):
        
        self.index = index_dict
        
        self.start=start_time
        
        self.end=end_time
        
        self.etf_names = list(self.index.values())
        
        try:
            data = yf.download(self.etf_names,start=start_time,end=end_time)
            
        except:
            reload(yf)
            data = yf.download(self.etf_names,start=start_time,end=end_time)
        
        adjclose = data['Adj Close']
        
        adjclose1 = adjclose.dropna() 
        
        self.data = adjclose1


    
    def select_etf(self, rolling_period, bench, num_asset_showing):
        
        P = sharpe.Screen(self.data,self.etf_names)

        Result = P.sharpe_ratio(rolling_period, bench)

        S = sharpe.Screen.pick_sharpe(Result,num_asset_showing)
        
        return S
    
    
    def Load_stock(self, etf_result, rolling_period, num_asset_showing, selected_etf=None):
                
               
        if selected_etf==None:
                   
            selected_etf=list(etf_result.index)
        
        for etf in selected_etf:
            
            names=self.index[etf]
            
            Stock=Ticker[names].values.tolist()
            
            Stock.append(etf)
            
            Stock_data = yf.download(Stock,start=self.start,end=self.end)
            
            P = sharpe.Screen(Stock_data, Stock)

            Result = P.sharpe_ratio(rolling_period, etf)

            S = sharpe.Screen.pick_sharpe(Result,num_asset_showing)
            
            print(S)
            
            
            
            
            
            
            
        
        
        
        
    















