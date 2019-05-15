# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:45:26 2019

@author: zfan01
"""
from vol import Covariance as cv

from vol import Predict_cov as pcv

from vol import vol_backtesting as vb

import scipy.optimize as opt 

from scipy.optimize import SR1

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

#Object is the current portfolio, data type as dataframe, col names as ticker name,
class Balance(object):

        
    
    #object function
    #object function
    def rosen(x,pre_cv):
            return x.dot(pre_cv).dot(x)
        
    def rosen1(x,pre_cv,re):
            mu=re.dot(x)
            var=x.dot(pre_cv).dot(x)
            return -mu/np.sqrt(var)
    
    def rosen2(x,pre_cv,re,target):
            mu=re.dot(x)
            return -mu
        
    
    def Optimizer(col_number, pre_cv, re, target,option=None):
        
        if option==None:
            
        
         #bonds of weights
                bds=opt.Bounds(np.zeros(col_number), np.repeat(np.inf,col_number))

                #linear constraints
                cnst=opt.LinearConstraint(np.array([np.ones(col_number),re]), [1,target], [1,np.inf])

                #initialize the weights
                x0=np.ones(col_number)/col_number

                #res=opt.minimize(rosen,x0, args=pre_cv, bounds=bds,constraints={cnst})

                res=opt.minimize(Balance.rosen,x0, args=pre_cv, method='trust-constr',  jac="2-point", hess=SR1(), bounds=bds,constraints=cnst)

        elif option==1:
            
                #bonds of weights
                bds=opt.Bounds(np.zeros(col_number), np.repeat(np.inf,col_number))

                #linear constraints    
                cnst=opt.LinearConstraint(np.array([np.ones(col_number)]), [1], [1])

                #initialize the weights
                x0=np.ones(col_number)/col_number

                #res=opt.minimize(rosen,x0, args=pre_cv, bounds=bds,constraints={cnst})
                res=opt.minimize(Balance.rosen1,x0, args=(pre_cv,re), method='trust-constr',  jac="2-point", hess=SR1(), bounds=bds,constraints=cnst)
                #res=opt.minimize(Balance.rosen1,x0, args=(pre_cv,re), method='SLSQP', bounds=bds,constraints=cnst)
        else:
                
                bds=opt.Bounds(np.zeros(col_number), np.repeat(np.inf,col_number))

                #cnst=opt.NonlinearConstraint(Balance.rosen,[0],[target])
                
                #nonlinear constraint
                cnst=({'type': 'ineq', 'fun': lambda x, pre_cv, re, target : target-x.dot(pre_cv).dot(x), 'args': (pre_cv,re,target)},
                      
                     {'type': 'eq', 'fun': lambda x, pre_cv, re, target : sum(x)-1, 'args': (pre_cv,re,target)},
                     )

                #initialize the weights
                x0=np.ones(col_number)/col_number


                res=opt.minimize(Balance.rosen2,x0, args=(pre_cv,re,target), method='SLSQP', bounds=bds,constraints=cnst)
                
                if np.sum(res.x)>2:
                    
                    res=opt.minimize(Balance.rosen2,x0, args=(pre_cv,re,target), method='trust-constr', bounds=bds,constraints=cnst)
                

                
        
        
        return res.x
        
             



    def find_weights(self,selected,target,timeoption):
        
        portfolio=self[self.columns[self.columns.isin(selected)]]
        
        
        info= cv.find_cov(portfolio,timeoption)
        
        #obtain the covaraince
        cov=info[0]
        
        #obtain the average return
        re= np.mean(info[2]).values
        
        #obtain the number of stocks
        col_number=info[2].shape[1]
        
        #obtain the forecasted covariance
        pre_cv=pcv.EWMA_cov(cov,0.5)
        
        
        #bonds of weights
        #bds=opt.Bounds(np.zeros(col_number), np.repeat(np.inf,col_number))
        
        #linear constraints
        #cnst=opt.LinearConstraint(np.array([np.ones(2),re]), [1,target], [1,np.inf])
        
        #initialize the weights
        #x0=np.ones(col_number)/col_number
        
        #res=opt.minimize(rosen,x0, args=pre_cv, bounds=bds,constraints={cnst})
        
        #res=opt.minimize(Balance.rosen,x0, args=pre_cv, method='trust-constr',  jac="2-point", hess=SR1(), bounds=bds,constraints=cnst)
    
        
        weights=Balance.Optimizer(col_number,pre_cv, re, target)
        
        
        return weights
    
    def find_shares(self, selected, weights,capital):
        
        money=capital*weights
        
        portfolio=self[self.columns[self.columns.isin(selected)]] 
        
        price=portfolio.iloc[0,:]
        
        shares=money/price
        #y = [2*a for a in x if a % 2 == 1]
        #[ x if x%2 else x*100 for x in range(1, 10) ]
        
        shares= [0 if x<1 else x for x in shares]
        
        Shares=pd.DataFrame({'Shares':shares,'Money':money},index=portfolio.columns)
        
        return Shares
        
        
        
        
    


    def back_testing(self, selected, capital, target, option1, bench, timeoption,option=None, period=None):
#        
#        #calculate the realized gain and loss
#     def back_error(self, option=None, period=None):
        
        if period == None:
        
            period=15
    
    #pick the selected columns
        portfolio=self[self.columns[self.columns.isin(selected)]]        
    
    # find the covaricance matrix
    
        bench_return=vb.back_error(self[[bench]],timeoption,option,period)[3]
    
        full=vb.back_error(portfolio,timeoption,option,period)
                   
        Pre_cov=full[1]
        
        Real_cov=full[2]
        
        Real_return=full[3]
        
        #take the average return
        re= np.mean(Real_return.iloc[:,]).values
        
     #L is the length of prediction period                                 
        L=len(Pre_cov)
        
        col_number=Real_return.shape[1]
                
        Error=[]
        
        Cap_Error=[]
        
        Weights=[]
        
        Pre_Capital=[capital]
        
        Real_Capital=[capital]
        
        Bench_Capital=[capital]
        
        Fake_Capital=[capital]
        
               
             
        for i in range(L,0,-1):
            
            past_return=Real_return.iloc[i,:].values
            
            realized_return=Real_return.iloc[i-1,:].values
            
            Bench_return=bench_return.iloc[i-1,:].values
                        
            #re= np.mean(Real_return.iloc[i:,]).values
            
            print(i)
            
            weights=Balance.Optimizer(col_number,Pre_cov[i-1], re, target,option1)
            
            fake_weights=Balance.Optimizer(col_number,Real_cov[i-1], re, target,option1)
            
            Weights.append(weights)
                        
#portfolio predicted returns
            Pre_return=np.sum(weights*past_return)
            
            Fake_return=np.sum(fake_weights*past_return)
            
#portfolio realized returns
            Realized_return=np.sum(weights*realized_return)
            
#bench portfolio returns
            

#calculate the prediction error                      
            Error.append(Pre_return-Realized_return)
            
            pre_capital_new=Real_Capital[-1]*(1+Pre_return)
            
            real_capital_new =Real_Capital[-1]*(1+Realized_return)
            
            bench_capital_new=Bench_Capital[-1]*(1+Bench_return)
            
            fake_capital_new=Fake_Capital[-1]*(1+Fake_return)
            
            
            Pre_Capital.append(pre_capital_new)
            
            Real_Capital.append(real_capital_new)
            
            Bench_Capital.append(bench_capital_new)
            
            Fake_Capital.append(fake_capital_new)
            
            Cap_Error.append(pre_capital_new-real_capital_new)
            
            
        Result=pd.DataFrame({'Real_Capital':Real_Capital[1:],
                              'Bench_Capital':Bench_Capital[1:],
                              'Pre_Capital':Pre_Capital[1:],
                              'Cap_Error':Cap_Error,
                              'Return_Error':Error})  
                
        
        list_of_datetimes=Real_return.index[:L].to_pydatetime()
        
        
        dates = matplotlib.dates.date2num(list_of_datetimes)

        #plt=matplotlib.pyplot.plot_date(dates, Error,'r--', dates, Real_Capital,'g--', dates, Pre_Capital,'b--', dates, Bench_Capital,'c--')
        matplotlib.pyplot.figure(1)
        plt.plot_date(dates, Error[::-1],'r--')
        plt.show()

        matplotlib.pyplot.figure(2)
        plt.plot_date(dates, Real_Capital[1:][::-1],'o--')  
        plt.plot_date(dates, Bench_Capital[1:][::-1],'c--')
        plt.plot_date(dates, Fake_Capital[1:][::-1],'r--')
        plt.show()
        
        matplotlib.pyplot.figure(3)
        plt.plot_date(dates, Real_Capital[1:][::-1],'g--')  
        plt.plot_date(dates, Pre_Capital[1:][::-1],'r--')
        plt.show()

        
        return [Result,Weights]
       


       
#        
#        
#        
#        
#        
#        
#    def adjustment(self, )
#        
#        
#        
#        
#        
#def num1(x):
#    y=2
#    def num2(y):
#        return x * y
#    return num2
#        
#        
#        
#    
#
#
#
#
#
#def main():
#    