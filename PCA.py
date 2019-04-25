#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:54:26 2018

@author: IanFan
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt


#calculatet the covariance of daily log_return
def corr_std(log_return):  
    X_std = StandardScaler().fit_transform(log_return)
    #to be consistant with the paper http://www.carolalexander.org/publish/download/JournalArticles/PDFs/OrthogonalGARCH_Primer.PDF
    X_std=X_std/np.sqrt(len(log_return))
    corr_mat=np.corrcoef(X_std.T)
    return [X_std,corr_mat]
#Obtain eigen value and eigen vectors
def eigen_return(log_return):
    [X_std,corr_mat]=corr_std(log_return)
    eig_vals, eig_vecs = np.linalg.eig(corr_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    return [X_std,eig_pairs]
#Obtain Principle return after the transformation
def PCA(log_return):
    [X_std,eig_pairs]=eigen_return(log_return)
    eig_mat=np.vstack([eig_pairs[i][1]]for i in range(len(eig_pairs))).T
    Pc=X_std.dot(eig_mat)
    return [Pc,eig_mat]

#using garch model to forecasts by inputing PCA  monthly return 
def vol_PCA_Inter(log_return):
    [Pc,eig_mat]=PCA(log_return)
    #transform back to dataframe with time index so that can be resampled into monthly data
    Pc_frame=pd.DataFrame(data=Pc, index=log_return.index)
   #obtain the monthly realized vol
    shape=log_return.shape
    forecasts=pd.DataFrame(data=np.zeros(shape),index=log_return.index + pd.DateOffset(months=1))
    #in sample and out of smaple forecast
    for i in range(0,shape[1]):
         garch11=arch_model(Pc_frame.iloc[:,i],p=1,q=1)
         res=garch11.fit(update_freq=10)
         forecasts.iloc[:,i]=(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + 
                  res.conditional_volatility**2 * res.params['beta[1]']).values
    
    return [forecasts,eig_mat]

#transform the Pc volatility back and obtain the final answer of vol   
def vol_PCA(log_return):
    [forecasts,eig_mat]=vol_PCA_Inter(log_return)
    sigma=np.diag(log_return.std(axis=0))
    A=eig_mat.dot(sigma)
    shape=forecasts.shape
    forecasts_cov=np.zeros((shape[1],shape[1],shape[0]))
    forecasts_var=np.zeros(shape)
    for i in range(0, shape[0]):
        D=np.diag(forecasts.iloc[i,:])
        forecasts_cov[:,:,i]=(A.dot(D).dot(A.T))*len(log_return)
        forecasts_var[i,:]=np.diagonal(forecasts_cov[:,:,i])
    return [forecasts_cov,forecasts_var]

#evaluate the realized monthly covariance
def realMcov(log_daily_return,log_return):
    shape1=log_daily_return.shape
    dindex=log_daily_return.index
    shape2=log_return.shape
    mindex=log_return.index
    realM_cov=np.zeros((shape2[1],shape2[1],shape2[0]))
    #loop to calculate the covariance
    for i in range(0,shape1[1]):
        #rtr is the logreturni*logreturnj
        rtr=np.multiply(log_daily_return.iloc[:,i:].values.T, log_daily_return.iloc[:,i].values.T).T
        RtR=pd.DataFrame(data=rtr,index=dindex)
        RtR_M=RtR.resample('M').sum()
        realM_cov[i,i:,:]=np.reshape(RtR_M,(1,shape2[1],shape2[0])
        realM_cov[i,i:,:]=np.reshape(RtR_M,(1,shape2[1],shape2[0])


def evaluate_vol(log_return):
    log_return_vol=log_daily_return.resample('M').std()
    realM_vol=log_return_vol*np.sqrt(log_return.resample('M').count())
  
    
#test code
data = yf.download(['AAPL','SPY','DOW','^IXIC'],start='2010-01-01',end='2018-01-01')
adjclose=data['Adj Close']
log_daily_return=np.log(adjclose.iloc[1:])-np.log(adjclose.iloc[:-1].values)
log_return=log_daily_return.resample('M').sum()
[forecasts_cov,forecasts_var]=vol_PCA(log_return)
realM_vol=(log_daily_return**2).resample('M').sum()




t1=np.arange(0,len(log_return),1)
t2=np.arange(0,len(log_return),1)
plt.plot(t1,forecasts_var[:,2],t2,realM_vol.iloc[:,2])
    
    
    
    
    









    