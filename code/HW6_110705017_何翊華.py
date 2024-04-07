import math
from sklearn import datasets
import numpy as np

def entropy(p1, n1):
    if p1==0 and n1==0:
        return 1
    elif p1==0 or n1==0:
        return 0
    else:
        pp = p1/(p1+n1)
        np = n1/(p1+n1)
        return -pp*math.log2(pp) - np*math.log2(np)
    
#print(entropy(5, 5))
#print(entropy(0, 10))
#print(entropy(2, 8))

def IG(p1, n1, p2, n2):
    pr = p1 + p2
    nr = n1 + n2
    num = pr + nr
    num1 = p1 + n1
    num2 = p2 + n2
    return entropy(pr, nr) - num1/num*entropy(p1, n1)-num2/num*entropy(p2, n2)

#print(IG(3, 4, 6, 1))
#print(IG(6, 2, 3, 3))

#print(IG(21, 5, 8, 30))
#print(IG(18, 33, 11, 2))

data = datasets.load_iris()

def ID3DTtrain(feature,target): # Each row have feature: A=? B=? C=? D=?, target: =0 or =1
    node = dict()
    node['data'] = range(len(target))
    tree = []
    tree.append(node)
    t = 0
    while t<len(tree):
        idx = tree[t]['data'] # update index list
        if sum(target[idx])==0: # all target = 0
            tree[t]['leaf']=1
            tree[t]['decision']=0
        elif sum(target[idx])==len(idx): # all target = 1
            tree[t]['leaf']=1
            tree[t]['decision']=1
        else:
            bestIG = 0
            # find best information gain 
            for i in range(feature.shape[1]): # check each feature (col)
                pool = list(set(feature[idx,i])) # select ith feature of every row and del duplicated values
                pool.sort()
                for j in range(len(pool)-1): # check values per 0.1
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if feature[k][i]<thres: # left child
                            G1.append(k)
                        else: # right child
                            G2.append(k)
                    p1 = sum(target[G1]==1)
                    n1 = sum(target[G1]==0)
                    p2 = sum(target[G2]==1)
                    n2 = sum(target[G2]==0)
                    thisIG = IG(p1,n1,p2,n2)
                    if thisIG>bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
                        
            # update current node and add nodes to tree
            if bestIG>0: # information gain > 0
                tree[t]['leaf'] = 0
                tree[t]['selectf'] = bestf
                tree[t]['threshold'] = bestthres
                tree[t]['child']=[len(tree),len(tree)+1]
                node = dict()
                node['data'] = bestG1
                tree.append(node) # add left node to tree
                node = dict()
                node['data'] = bestG2 # add right node to tree
                tree.append(node)
            else: # information gain = 0
                tree[t]['leaf'] = 1
                if sum(target[idx]==1)>sum(target[idx]==0):
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
        while(Tree[t]['leaf']==0):
            node = Tree[t]
            if(feature1[node['selectf']]<node['threshold']):
                t = node['child'][0]
            else:
                t = node['child'][1]
        target[i] = Tree[t]['decision']
    return target

# 1.

# data [row][features]  
# target [row][label(1 or 2)]

# training                   
feature1 = data.data[50:80][:]
feature2 = data.data[100:130][:]
feature = np.concatenate((feature1,feature2),axis=0)
target = np.concatenate((data.target[50:80]-1,data.target[100:130]-1))
T = ID3DTtrain(feature,target)

# training accuracy
predtarget = ID3DTtest(T,feature)
training_accuracy = np.sum(predtarget==target)/len(target)
print("1. training accuracy: ",training_accuracy)

# testing
feature1_test = data.data[80:100][:]
feature2_test = data.data[130:150][:]
feature_test = np.concatenate((feature1_test,feature2_test),axis=0)
target_test = np.concatenate((data.target[80:100]-1,data.target[130:150]-1))

# testing accuracy
predtarget_test = ID3DTtest(T,feature_test)
testing_accuracy = np.sum(predtarget_test==target_test)/len(target_test)
print("1. testing accuracy: ",testing_accuracy)

# 2. 

# training

feature0 = data.data[0:30][:]
feature1 = data.data[50:80][:]
feature2 = data.data[100:130][:]
target0 = data.target[0:30]
target1 = data.target[50:80]
target2 = data.target[100:130]

feature_train = np.concatenate((feature0, feature1, feature2),axis=0)
target_train = np.concatenate((target0, target1, target2), axis=0)

# label(0 or 1)
feature01 = np.concatenate((feature0,feature1),axis=0)
target01 = np.concatenate((target0,target1))
T01 = ID3DTtrain(feature01,target01)
# label(1 or 2)
feature12 = np.concatenate((feature1,feature2),axis=0)
target12 = np.concatenate((target1-1,target2-1))
T12 = ID3DTtrain(feature12,target12)
# label(0 or 1)
feature02 = np.concatenate((feature0,feature2),axis=0)
target02 = np.concatenate((target0,target2-1))
T02 = ID3DTtrain(feature02,target02)

# training accuracy
predtarget01 = ID3DTtest(T01,feature_train) # 0 -> 0, 1 -> 1
print(predtarget01)
predtarget12 = ID3DTtest(T12,feature_train) # 0 -> 1, 1 -> 2
for i in range(len(predtarget12)):
    predtarget12[i]+=1
print(predtarget12)
predtarget02 = ID3DTtest(T02,feature_train) # 0 -> 0, 1 -> 2
for i in range(len(predtarget02)):
    if predtarget02[i] == 1:
        predtarget02[i] = 2
print(predtarget02)

# voting
N = feature_train.shape[0]
predtarget_train = np.zeros((N,))
for i in range(90):
    count = [0, 0, 0]
    count[int(predtarget01[i])]+=1
    count[int(predtarget12[i])]+=1
    count[int(predtarget02[i])]+=1
    imax = np.array(count).argmax()
    predtarget_train[i] = imax
    
print(predtarget_train)
training_accuracy_2 = np.sum(predtarget_train==target_train)/len(target_train)
print("2. training accuracy: ",training_accuracy_2)

# testing

feature0_test = data.data[30:50][:]
feature1_test = data.data[80:100][:]
feature2_test = data.data[130:150][:]
target0_test = data.target[30:50]
target1_test = data.target[80:100]
target2_test = data.target[130:150]

feature_test = np.concatenate((feature0_test, feature1_test,feature2_test),axis=0)
target_test = np.concatenate((target0_test, target1_test, target2_test))

# testing accuracy
predtarget01_test = ID3DTtest(T01,feature_test) # 0 -> 0, 1 -> 1
#print(predtarget01_test)
predtarget12_test = ID3DTtest(T12,feature_test) # 0 -> 1, 1 -> 2
for i in range(len(predtarget12_test)):
    predtarget12_test[i]+=1
#print(predtarget12_test)
predtarget02_test = ID3DTtest(T02,feature_test) # 0 -> 0, 1 -> 2
for i in range(len(predtarget02_test)):
    if predtarget02_test[i] == 1:
        predtarget02_test[i] = 2
#print(predtarget02_test)

#voting
N = feature_test.shape[0]
predtarget_test = np.zeros((N,))
for i in range(60):
    count = [0, 0, 0]
    count[int(predtarget01_test[i])]+=1
    count[int(predtarget12_test[i])]+=1
    count[int(predtarget02_test[i])]+=1
    imax = np.array(count).argmax()
    predtarget_test[i] = imax
    
#print(predtarget_test)
testing_accuracy_2 = np.sum(predtarget_test==target_test)/len(target_test)
print("2. testing accuracy: ",testing_accuracy_2)

