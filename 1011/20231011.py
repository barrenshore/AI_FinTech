# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:20:54 2023

@author: USER
"""
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def BLScall(S,K,T,r,vol):
    d1 = (math.log(S/K)+(r+vol*vol/2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call = S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
    return call

S = 16273.38
K = 16500
T = 14/365
r = 0.0166
vol = 0.142
call = 93.0

def BisectionBLS(S,K,T,r,call):
    left = 0.000001
    right = 1
    while(right-left>0.00001):
        middle = (left+right)/2
        if(BLScall(S,K,T,r,middle)-call)*(BLScall(S,K,T,r,left)-call)<0:
            right = middle
        else:
            left = middle
    return (left+right)/2

def MCsim(S,T,r,vol,N,M):
    dt = T/N
    St = np.zeros((M,N+1))
    St[:,0] = S
    for i in range(N):
        St[:,i+1] = np.multiply(St[:,i],np.exp((r-0.5*vol*vol)*dt+vol*np.random.normal(0,1,(M,))*math.sqrt(dt)))
    return St
    
vol = BisectionBLS(S,K,T,r,call)    
print(vol)
print(BLScall(S,K,T,r,vol))
St = MCsim(S,T,r,vol,100,1000)
call1 = np.mean(np.maximum(St[:,-1]-K,0))*math.exp(-r*T)
plt.plot(St.transpose())
print(call1)










