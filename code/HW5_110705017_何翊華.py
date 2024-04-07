import math
import numpy as np
from scipy.stats import norm

def BLScall(S, K, T, r, vol):
    d1 = (math.log(S/K) + (r + vol*vol/2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return call

def bitcall(S0, K, T, r, vol, N):
    dt = T/N
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    p = (math.exp(r*dt)-d)/(u-d)
    STree = np.zeros((N+1, N+1))
    PTree = np.zeros((N+1, N+1))
    
    STree[0][0] = S0
    for c1 in range(N):
        STree[0][c1+1] = STree[0][c1]*u
        for r1 in range(c1+1):
            STree[r1+1][c1+1] = STree[r1][c1]*d
            
    PTree[0][0] = 1
    for c1 in range(N):
        for r1 in range(c1+1):
            PTree[r1][c1+1] += PTree[r1][c1]*p
            PTree[r1+1][c1+1] += PTree[r1][c1]*(1-p)
            
    call = 0
    for r1 in range(N+1):
        if STree[r1][N]>=K:
            call += (STree[r1][N]-K)*PTree[r1][N]
            
    return call*math.exp(-r*T)

#1
print(BLScall(16273.38, 16500, 14/365, 0.0166, 0.14134)-93)

#2
for n in [10, 100, 1000]:
    print(bitcall(16273.38, 16500, 14/365, 0.0166, 0.14134, n)-93)
  
#3
dS = 0.00001
callS = BLScall(16273.38, 16500, 14/365, 0.0166, 0.14134)
callS_dS = BLScall(16273.38+dS, 16500, 14/365, 0.0166, 0.14134)
print((callS_dS-callS)/dS)