import numpy as np
import matplotlib.pyplot as plt

# Regression

def F1(t):
    return 0.063*(t**3)-5.284*(t**2)+4.887*t+412+np.random.normal(0,1)

def F2(t,A,B,C,D):
    return A*(t**B)+C*np.cos(D*t)+np.random.normal(0,1,t.shape)

def E(b2,A2,A,B,C,D):
    return np.sum(abs(F2(A2,A,B,C,D)-b2))

n = 1000
# shape (row, col)
b1 = np.zeros((n,1)) 
A1 = np.zeros((n,5))
for i in range(n):
    # np.random.random() => [0, 1)
    t = np.random.random()*100 # generate t from interval [0, 100)
    b1[i] = F1(t)
    A1[i,0] = t**4
    A1[i,1] = t**3
    A1[i,2] = t**2
    A1[i,3] = t
    A1[i,4] = 1
x = np.linalg.lstsq(A1,b1)[0] # A1 x = b1
print(x)

A2 = np.random.random((1000,1))*100 # generate 1000 ele from interval [0, 100)
b2 = F2(A2,0.6,1.2,100,0.4)

px1 = np.zeros((1024,2))
for i in range(1024):
    D = (i-511)/100
    px1[i,0] = D
    px1[i,1] = E(b2,A2,0.6,1.2,100,D)
plt.plot(px1[:,0],px1[:,1])

# px2 = np.zeros((1024,1024,3))
# for i in range(1024):
#     A = (i-511)/100
#     for j in range(1024):
#         C = j-511
#         px2[i,j,0]=A
#         px2[i,j,1]=C
#         px2[i,j,2]=E(b2,A2,A,1.2,C,0.4)

#%%

# Genetic Algorithm

p = 10000 # population
r = 0.01 # survival_rate
m = 1000 # mutation_rate
g = 10 # generation
survive = round(p*r)
pop = np.random.randint(0,2,(p,40))
fit = np.zeros((p,1))
for generation in range(g):
    for i in range(p):
        gene = pop[i,:]
        A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100
        B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
        C = np.sum(2**np.array(range(10))*gene[20:30])-511
        D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
        fit[i] = E(b2,A2,A,B,C,D)
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
    for i in range(survive,p):
        fid = np.random.randint(0,survive)
        mid = np.random.randint(0,survive)
        while mid==fid:
            mid = np.random.randint(0,survive)
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:].copy()
        father = pop[fid,:]
        son[mask[0,:]==1] = father[mask[0,:]==1]
        pop[i,:]=son
    for i in range(m):
        mr = np.random.randint(survive,p)
        mc = np.random.randint(0,40)
        pop[mr,mc] = 1-pop[mr,mc]

for i in range(p):
    gene = pop[i,:]
    A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100
    B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
    C = np.sum(2**np.array(range(10))*gene[20:30])-511
    D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
    fit[i] = E(b2,A2,A,B,C,D)
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]

gene = pop[0,:]
A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100
B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
C = np.sum(2**np.array(range(10))*gene[20:30])-511
D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
print('A:',A,' B:',B,' C:',C,' D:',D)    
    
    