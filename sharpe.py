#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:23:40 2019

@author: IanFan
"""

import vol as cov

import pandas as pd

import numpy as np

class Screen(object):
    
    def __init__(self, data, selected):
            
        portfolio=data[data.columns[data.columns.isin(selected)]]
        
        self.portfolio=portfolio
        
        self.asset_num=portfolio.shape[1]
        
        self.length=portfolio.shape[0]
        
        
        
    def sharpe_ratio(self,timeoption,bench):
        
        portfolio=self.portfolio
               
        asset_num=self.asset_num
        
        Sharpe=pd.DataFrame(columns=portfolio.columns)
        
        for (asset,n) in zip(portfolio.columns,range(asset_num)):
            
            if asset==bench:
                
                unit=portfolio[bench]
                
                repeat = portfolio[bench].values
                
                portfolio1=portfolio.copy()
                
                portfolio1=portfolio1.assign(SPY1=repeat)
                
                unit=portfolio1[[bench,'SPY1']]
                
            else:
                selected_unit=[bench,asset]
                
                unit=portfolio[portfolio.columns[portfolio.columns.isin(selected_unit)]]
            
            unit_info=cov.Covariance.find_cov(unit,timeoption)
            
            unit_cov=unit_info[0]
            
            unit_cov=np.array(unit_cov)
            
            unit_return=unit_info[2]
            
            diff=unit_return[asset]-unit_return[bench]
            
            #i is the index of the chosen asset in the dataset
            i=unit_return.columns.values.tolist().index(asset)
            
            
            #since the shape of data as been changed, need to adjust the index
            if i == 1:
                i=3
            
            #initialize the correlation matrix
            if n == 0:               
                
                L=len(unit_cov)
                
                Full_corr=np.zeros((L,asset_num))
                
                Full_sharpe=np.zeros((L,asset_num))
                
                Index=diff.index
                
            #reshape the unit_cov so that it can be easy to be used
            
            unit_cov=unit_cov.reshape((L,1,4))   
            
            try:
                full_corr = unit_cov[:,:,1]/np.sqrt(unit_cov[:,:,0]*unit_cov[:,:,3])
                
                full_stv= np.sqrt(unit_cov[:,:,i].reshape(1,L))
                
                full_sharpe = diff.values/full_stv
                
            except:
                full_corr = np.float64(unit_cov[:,:,1])/np.sqrt(unit_cov[:,:,0]*unit_cov[:,:,3])

                full_stv= np.sqrt(unit_cov[:,:,i].reshape(1,L))
                
                full_sharpe = diff.values/full_stv
            
            
            full_corr = full_corr.reshape(L)
            
            Full_corr[:,n]=full_corr
            
            Full_sharpe[:,n]=full_sharpe
            
            corr = np.mean(full_corr[full_corr!=np.inf])
            
            sharpe=np.mean(full_sharpe[full_sharpe!=np.inf])
            
            stv=np.mean(full_stv[full_stv!=np.inf])
            
            Sharpe[asset]=[sharpe,corr,stv]
            
            
        Sharpe.index=['Sharpe','Correlation','Volatility']
        
        Full_Sharpe=pd.DataFrame(Full_sharpe,columns=portfolio.columns,index=Index)
        
        Full_Corr=pd.DataFrame(Full_corr,columns=portfolio.columns,index=Index)
        
        return [Sharpe, Full_Sharpe, Full_Corr]
    
    #result is the [Sharpe, Full_Sharpe, Full_Corr]
    def pick_sharpe(result,num):
        
        Sharpe=result[0].T
        
        Sharpe=Sharpe.sort_values(by='Sharpe',ascending=False)
        
        if num < len(Sharpe):
            
            Sharpe=Sharpe.iloc[0:num,:]
        
        Sharpe=Sharpe.sort_values(by='Volatility',ascending=True)
        
        return Sharpe
    
    #def show_sharpe(result):
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
    