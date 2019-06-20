#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:39:03 2019

@author: IanFan
"""
import numpy as np


#self is the log(first component) of find cov in vol.py
class HAR(object):
    
    def __init__(self,asset_size,windows,har):
        
        self.asset_size= asset_size
        self.windows= windows
        self.har=har

        
    def flat_matrix(self):
        
        cov_period=self[:self.windows]
        
        flat_cov=np.array([cov[np.triu_indices(self.asset_size)] for cov in cov_period])
        
        return flat_cov
    
    def rolling_window(a,window):
        
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        
        strides = a.strides + (a.strides[-1],)
        
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def create_features(self):
        
        [har1,har2,har3]=self.har
        
        flat_cov=HAR.flat_matrix(self)
        
        full = HAR.rolling_windows(flat_cov,har3)
        
        feature3=full[:,:,1:har3].sum(axis=2)
        
        feature2=full[:,:,1:har2].sum(axis=2)
        
        feature1=full[:,:,1:har1].sum(axis=2)
        
        response=full[:,:,0].sum(axis=2)
        
        return [response,feature1,feature2,feature3]
    
    def rosen(data,beta):
        
        response=data[0]
        
        feature1=data[1]
        
        feature2=data[2]
        
        feature3=data[3]
        
        size=len(reponse[0])
        
        predicted=beta[0:size]+feature1*beta[size:2*size]+feature2*beta[2*size:3*size]+feature
        
        y=np.sum((predicted-reponse)**2)
        
        
    
    
    
    def optimize_ols(self):
        
        #response=b0+b1*feature1+b2*feautre2+b3*feature3+epsi
        
        
        
        
        
    
    
    
        
        
        
        
        
        
    