# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:15:22 2023

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt
CALLprice = {16100:293,16200:236,16300:177,16400:131,16500:93,16600:64,16700:43.5,16800:28.5}
PUTprice = {16100:114,16200:149,16300:194,16400:246,16500:308,16600:384,16700:466,16800:555}
ST = np.arange(15800,17200)

def OptionProfit(pos):
    callposition = pos[0]
    putposition = pos[1]
    profit = np.zeros(ST.shape)
    for position in callposition:
        K = position[0]
        N = position[1]
        C = CALLprice[K]
        profit = profit+(np.maximum(ST-K,0)-C)*N
    for position in putposition:
        K = position[0]
        N = position[1]
        P = PUTprice[K]
        profit = profit+(np.maximum(K-ST,0)-P)*N
    return profit

pos1 = [[[16200,1]],[]]
p1 = OptionProfit(pos1)
pos2 = [[[16200,1],[16800,-1]],[]]
p2 = OptionProfit(pos2)
pos3 = [[],[[16200,-1],[16800,-1],[16500,2]]]
p3 = OptionProfit(pos3)
plt.plot(ST,p1,'b-',ST,p2,'r-',ST,p3,'g-',[ST[0],ST[-1]],[0,0],'k--')
        
        
        
        
        
        
        
        
        
        