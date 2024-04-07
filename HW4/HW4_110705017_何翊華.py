import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def BLScall(S, K, T, r, vol):
    d1 = (math.log(S/K) + (r + vol*vol/2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return call

def BisectionBLS(S, K, T, r, call):
    left = 0.000001
    right = 1
    while right - left > 0.00001:
        middle = (left+right)/2
        if (BLScall(S, K, T, r, middle) - call)*(BLScall(S, K, T, r, left) - call) < 0:
            right = middle
        else:
            left = middle
    return (left+right)/2

def MCsim(S, T, r, vol, N, M): # N time slices, M trials
    dt = T/N
    St = np.zeros((M, N+1)) # M row, N+1 col
    St[:, 0] = S # row, col
    for i in range(N):
        St[:,i+1] = np.multiply(St[:,i],np.exp((r-0.5*vol*vol)*dt+vol*np.random.normal(0,1,(M,))*math.sqrt(dt)))
    return St
    
#1-1
S = 16273.38
K = 16500
T = 14/365
r = 0.0166

call = 93
vol = BisectionBLS(S, K, T, r, call)
print(vol)
print(BLScall(S, K, T, r, vol))
print()


#1-2
Sum = 0
for i in range(10):
    St = MCsim(S, T, r, vol, 100, 1000)
    calls = np.mean(np.maximum(St[:, -1] - K, 0))*math.exp(-r*T)
    Sum+=calls
    
Avg = Sum/10
error = abs(Avg-call)
print("avg error: ", error)
print()


#1-3
for trials in [100, 1000, 10000]:
    Sum = 0
    for i in range(10):
        St = MCsim(S, T, r, vol, 100, trials)
        calls = np.mean(np.maximum(St[:, -1] - K, 0))*math.exp(-r*T)
        Sum+=calls
    Avg = Sum/10
    error = abs(Avg-call)
    print(trials, " trials error: ", error)

print()
for slices in [10, 100, 1000, 5000]:
    Sum = 0
    for i in range(10):
        St = MCsim(S, T, r, vol, slices, 1000)
        calls = np.mean(np.maximum(St[:, -1] - K, 0))*math.exp(-r*T)
        Sum+=calls
    Avg = Sum/10
    error = abs(Avg-call)
    print(slices, " slices error: ", error)
print()


#2
K_list = [16100, 16200, 16300, 16400, 16500, 16600, 16700, 16800]
call_list = [293, 236, 177, 131, 93, 64, 43.5, 28.5]
vol_list = []

for i in range(8):
    call_i = call_list[i]
    K_i = K_list[i]
    vol = BisectionBLS(S, K_i, T, r, call_i)
    vol_list.append(vol)
    
print(vol_list)
xpoints = np.array(K_list) # X: price
ypoints = np.array(vol_list) # Y: volatility

plt.plot(xpoints, ypoints)
plt.show()
    




