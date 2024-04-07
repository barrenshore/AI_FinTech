# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:48:04 2023

@author: USER
"""
import math
from sklearn import datasets
import numpy as np

def entropy(p1,n1):
    if(p1==0 and n1==0):
        return 1
    elif(p1==0 or n1==0):
        return 0
    else:
        pp = p1/(p1+n1)
        np = n1/(p1+n1)
        return -pp*math.log2(pp) - np*math.log2(np)

def IG(p1,n1,p2,n2):
    pr = p1 + p2
    nr = n1 + n2
    num = pr+nr
    num1 = p1+n1
    num2 = p2+n2
    return entropy(pr,nr)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)
    

print(entropy(5,5))
print(entropy(10,0))
print(entropy(2,8))
print(IG(3,4,6,1))
print(IG(6,2,3,3))
print(IG(21,5,8,30))
print(IG(18,33,11,2))

data = datasets.load_iris()

def ID3DTtrain(feature,target):
    node = dict()
    node['data'] = range(len(target))
    tree = []
    tree.append(node)
    t = 0
    while(t<len(tree)):
        idx = tree[t]['data']
        if(sum(target[idx])==0):
            tree[t]['leaf']=1
            tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            tree[t]['leaf']=1
            tree[t]['decision']=1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k][i]<thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    p1 = sum(target[G1]==1)
                    n1 = sum(target[G1]==0)
                    p2 = sum(target[G2]==1)
                    n2 = sum(target[G2]==0)
                    thisIG = IG(p1,n1,p2,n2)
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                tree[t]['leaf'] = 0
                tree[t]['selectf'] = bestf
                tree[t]['threshold'] = bestthres
                tree[t]['child']=[len(tree),len(tree)+1]
                node = dict()
                node['data'] = bestG1
                tree.append(node)
                node = dict()
                node['data'] = bestG2
                tree.append(node)
            else:
                tree[t]['leaf']=1
                if(sum(target[idx]==1)>sum(target[idx]==0)):
                    tree[t]['decision'] = 1
                else:
                    tree[t]['decision'] = 0
        t+=1
    return tree

def ID3DTtest(Tree,feature):
    N = feature.shape[0]
    target = np.zeros((N,))
    for i in range(N):
        feature1 = feature[i,:]
        t = 0
        while(T[t]['leaf']==0):
            node = T[t]
            if(feature1[node['selectf']]<node['threshold']):
                t = node['child'][0]
            else:
                t = node['child'][1]
        target[i] = T[t]['decision']
    return target
        
feature1 = data.data[50:80][:]
feature2 = data.data[100:130][:]
feature = np.concatenate((feature1,feature2),axis=0)
target = np.concatenate((data.target[50:80]-1,data.target[100:130]-1))

T = ID3DTtrain(feature,target)
predtarget = ID3DTtest(T,feature)
training_accuracy = np.sum(predtarget==target)/len(target)
print(training_accuracy)







