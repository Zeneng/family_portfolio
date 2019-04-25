#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:09:36 2018

@author: IanFan
"""
import numpy as np

import pandas as pd

#from datetime import timedelta
'Variable is stock adj close'

import matplotlib

#object is a data frame,last five years adjust close price with days as index
#ex=pd.read_csv('etf_example.csv',parse_dates=['Date'], dayfirst=True,index_col=0)
#
#
#
#
#
#
#log_return=np.log(ex.iloc[1:,:])-np.log(ex.iloc[:-1,:].values)
#
#
#
#monthly_vol=log_return.resample('M').std()*np.sqrt(21)
#
#month=log_return.resample('M').mean()
#
#M=month.index
#
#cov_dic=dict.fromkeys(M)
#for m in M:
#    r=log_return.loc[m-1:m]
#    cov_dic[m]=np.cov(r.T)*21
    


#object is a data frame with datetime as index, and ticker name as col names
class Covariance(object):
    
    
    def daily_return(self):
        simple_return=self.iloc[1:,:]/self.iloc[:-1,:].values-1
        return simple_return
    
    def monthly_time(self):
        Time_list=[]
        time=self.index[-1]
        i=-1
        while time > self.index[0]:
            try:
                Time_list.append(time)
                i=i-21
                time=self.index[i]
            #time=time-timedelta(days=30)
            except:
                return Time_list
        
                

    def daily_logreturn(self):
        log_return=np.log(self.iloc[1:,:])-np.log(self.iloc[:-1,:].values)
        return log_return
    

    def find_vol(self):
        daily_log_return = Covariance.daily_logreturn(self)
        daily_log_return_var=daily_log_return.resample('M').var()
        Monthly_var=daily_log_return_var*21
       
        return Monthly_var

    def find_cov(self):
        
        daily_log_return = Covariance.daily_logreturn(self)
        
        daily_return = Covariance.daily_return(self)
        
                
        #Month_return=daily_log_return.resample('M').sum()
        
        Month = Covariance.monthly_time(self)
        
        Month1=Month[:-1]
        
        Month2=Month[1:]
        
        
        
        #initialize the cov_dic
        cov_dic=dict.fromkeys(Month1)
        #initialize the covariance
        cov_list=[]
        
        Month_return=pd.DataFrame(index=Month1,columns=self.columns)
        
        
        for (m1, m2) in zip(Month1, Month2):
            r=daily_log_return.loc[m2:m1]
            
            r1=daily_return.loc[m2:m1] +1
            
            Month_return.loc[m1] = r1.product(axis=0)  - 1                   
            
            cov=np.cov(r.T)*21
            
            cov_dic[m1]=cov
            
            cov_list.append(cov)
        
        cov_list=np.asarray(cov_list)
            
        return [cov_list, cov_dic, Month_return]
    

#object here is np array of arrays
class Predict_cov(object):

#weigt is a number between 0 and 1
    
    
    def EWMA_cov(self, weight=None, period=None):
        
        if period==None:
            
            period=len(self)
        
        if weight==None:
            
            weight=0.5
                
        weights = np.geomspace(1, weight**(period-1),num=period)
        
        train=self[-period:]
        
        if len(self.shape)>1:
                
            EWMA = sum(train*weights.reshape(period,1,1))/ np.sum(weights)
        else:
            EWMA = sum(train*weights)/ np.sum(weights)
        
        return EWMA
    
    #def HAR(self,)

    #def GARCH(self,)
    
    def Pred(self, weight=None, period=None, option=None):
        
        if option==None:
            return Predict_cov.EWMA_cov(self, weight, period)
        
    
        
#need to adopt HAR model
#multi dimensional GARCH model

#the object is a long period of data. Need to have a for loop go through it
class vol_backtesting(object):

  
    def back_error(self, option=None, period=None):
        
        if period ==None:
            
            period=20
            
# find the covaricance matrix
        full=Covariance.find_cov(self)
        
                
        full_cov=full[0]
        
        
        Month_return=full[2]
        
               
        L=len(full_cov)
        
        end=L-period
        
        Error=[]
        
        Pre=[]
        
        Real=[]
#loop to find all the prediction error        
        for k in range(1,end+1):
            
            real=full_cov[k-1]
            
            cov=full_cov[k:k+period]
            
            pre=Predict_cov.Pred(cov,option)
            
            Error.append(np.sum(abs(pre-real)))
            
            Pre.append(pre)
            
            Real.append(real)
                
        
    #plot the error if it's for portfolio purpose, turned off if this for     
        if self.shape[1]>1:
            
            list_of_datetimes=Month_return.index[0:end].to_pydatetime()
            
            
            dates = matplotlib.dates.date2num(list_of_datetimes)
    
            
            matplotlib.pyplot.plot_date(dates, Error)
                    
            
        return [Error,Pre,Real,Month_return]
       
    
    
        
    
    
        
  
            
            
            
            
            
            
            
            
        
            
        
        
        
        
        
        
    


#object is the period of 
    
    
            
        
        
        
        




    
