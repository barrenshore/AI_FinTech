from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def F(t, A, B, C, tc, w, fi, beta): 
    return A + B*np.power(tc-t,beta) + B*C*np.power(tc-t,beta)*np.cos(w*np.log(tc-t)+fi)

# Load data
data = pd.read_csv('D:/人工智慧/AAPL.csv')

# Convert time to ordinal
time = [pd.Timestamp.toordinal(dt.strptime(t1, '%Y-%m-%d')) for t1 in data['Date']]

# Time interval for tc
ts = pd.Timestamp(year = 2023,  month = 7, day = 24) #738725
te = pd.Timestamp(year = 2023,  month = 8, day = 4) #738736
print("tc range from ", ts.toordinal(), " to ", te.toordinal())

# Create list of ln(adj_price)
price = np.log(data['Adj Close'].values)

# Initialize
N = 10000 # population
G = 5 # generation
s = 0.05 # survival_rate
m = 0.01 # mutation_rate
n = 138 # time interval
survive = round(N*s)
mutate = round(N*m)

pop = np.random.randint(0,2,(N,30)) # generate 30 bits of 0 or 1 for N times
fit = np.zeros((N,1)) # generate 0 for N times

# Genetic algorithm
for generation in range(G):
    print("generation: ", generation) #debug
    for i in range(N):
        gene = pop[i,:]
        
        # Range for tc, w, fi, beta
        w = np.sum(2**np.array(range(10))*gene[:10])/50 # [0, 1024] => [0, 20.48] 
        beta = np.sum(2**np.array(range(10))*gene[10:20])/1000 # [0, 1024] => [0, 1.024] 
        fi = np.sum(2**np.array(range(6))*gene[20:26])/10 # [0, 1024] => [0, 6.4] 
        tc = np.sum(2**np.array(range(4))*gene[26:30])+738725 # [0, 16] => [738725, 738740]
        
        # Linear regression for A, B, C
        b = price[:n]
        a = np.zeros((n,3))
        for j in range(n):
            a[j,0] = 1
            a[j,1] = np.power(tc-time[j],beta)
            a[j,2] = np.power(tc-time[j],beta)*np.cos(w*np.log(tc-time[j]))
        x = np.linalg.lstsq(a,b)[0]
        A = x[0]
        B = x[1]
        C = x[2]/B
        t0 = range(n)
        fit[i] = np.mean(((A+B*a[t0,1]+B*C*a[t0,2])-b)**2) 
    
    sortf = np.argsort(fit[:,0]) # indices of sorted + transposed fit matrix
    pop = pop[sortf,:] # sort pop by indices
    
    # Survival
    for i in range(survive,N): # index from survive to N-1
        fid = np.random.randint(0,survive) # father's index
        mid = np.random.randint(0,survive) # mother's index
        while mid==fid:
            mid = np.random.randint(0,survive)
        mask = np.random.randint(0,2,(1,30)) # create filter
        son = pop[mid,:].copy()
        father = pop[fid,:]
        son[mask[0,:]==1] = father[mask[0,:]==1] # interchange bits
        pop[i,:]=son # change ith row to son

    # Mutation
    for i in range(mutate):
        mr = np.random.randint(survive,N) # choose a row
        mc = np.random.randint(0,30) # choose a bit
        pop[mr,mc] = 1-pop[mr,mc] # change bit

# Compute last generation's MSE
for i in range(N):
    gene = pop[i,:]
    
    # Range for tc, w, fi, beta
    w = np.sum(2**np.array(range(10))*gene[:10])/50 # [0, 1024] => [0, 20.48]
    beta = np.sum(2**np.array(range(10))*gene[10:20])/1000 # [0, 1024] => [0, 1.024] 
    fi = np.sum(2**np.array(range(6))*gene[20:26])/10 # [0, 1024] => [0, 6.4] 
    tc = np.sum(2**np.array(range(4))*gene[26:30])+738725 # [0, 16] => [738725, 738740]
    
    # Linear regression for A, B, C
    b = price[:n]
    a = np.zeros((n,3))
    for j in range(n):
        a[j,0] = 1
        a[j,1] = np.power(tc-time[j],beta)
        a[j,2] = np.power(tc-time[j],beta)*np.cos(w*np.log(tc-time[j]))
    x = np.linalg.lstsq(a,b)[0]
    A = x[0]
    B = x[1]
    C = x[2]/B
    t0 = range(n)
    fit[i] = np.mean(((A+B*a[t0,1]+B*C*a[t0,2])-b)**2) 
    
sortf = np.argsort(fit[:,0]) # indices of sorted + transposed fit matrix
pop = pop[sortf,:] # sort pop by indices




# Fit the model by best gene

gene = pop[0,:] # choose the best gene

# Range for tc, w, fi, beta
w = np.sum(2**np.array(range(10))*gene[:10])/50 # [0, 1024] => [0, 20.48] 
beta = np.sum(2**np.array(range(10))*gene[10:20])/1000 # [0, 1024] => [0, 1.024] 
fi = np.sum(2**np.array(range(6))*gene[20:26])/10 # [0, 1024] => [0, 6.4] 
tc = np.sum(2**np.array(range(4))*gene[26:30])+738725 # [0, 16] => [738725, 738740]

# Linear regression for A, B, C
b = price[:n]
a = np.zeros((n,3))
for j in range(n):
    a[j,0] = 1
    a[j,1] = np.power(tc-time[j],beta)
    a[j,2] = np.power(tc-time[j],beta)*np.cos(w*np.log(tc-time[j]))
x = np.linalg.lstsq(a,b)[0]
A = x[0]
B = x[1]
C = x[2]/B

print("A: ", A)
print("B: ", B)
print("C: ", C)
print("tc: ", datetime.date.fromordinal(tc))
print("w: ", w)
print("fi: ", fi)
print("beta: ", beta)

# Draw graph
price_pos = []
for i in range(n):
    price_pos.append(F(time[i], A, B, C, tc, w, fi, beta))
price_pos = np.array(price_pos)

date = []
for i in range(n):
    date.append(datetime.date.fromordinal(time[i]))
date = np.array(date)

# Plot lines 
plt.figure(figsize=(20,10))
plt.plot(date[:n], np.exp(price[:n]), label = "Original Price") 
plt.plot(date[:n], np.exp(price_pos), label = "Predict Price") 
plt.legend()
plt.show()

fit_best = np.mean((np.exp(price[:n])-np.exp(price_pos))**2)
print("MSE: ", fit_best)
