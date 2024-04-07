import numpy as np
import matplotlib.pyplot as plt

CALLprice = {16100:293, 16200:236, 16300:177, 16400:131, 16500:93, 16600:64, 16700:43.5, 16800:28.5}
PUTprice = {16100:114, 16200:149, 16300:194, 16400:246, 16500:308, 16600:384, 16700:466, 16800:555}
ST = np.arange(15800, 17200)

def OptionProfit(pos):
    callposition = pos[0]
    putposition = pos[1]
    profit = np.zeros(ST.shape)
    for position in callposition:
        K = position[0]
        N = position[1]
        C = CALLprice[K]
        profit = profit+(np.maximum(ST-K, 0)-C)*N
    for position in putposition:
        K = position[0]
        N = position[1]
        P = PUTprice[K]
        profit = profit+(np.maximum(K-ST, 0)-P)*N
    return profit

#1

#butterfly spread x2
pos_butter_1 = [[[16300, -1], [16700, -1], [16500, 2]], []]
pos_butter_2 = [[], [[16200, -1], [16800, -1], [16500, 2]]]
pb1 = OptionProfit(pos_butter_1)
pb2 = OptionProfit(pos_butter_2)

#straddle x1
pos_straddle = [[[16500, 1]], [[16500, 1]]]
psd = OptionProfit(pos_straddle)

#strangle x2
pos_strangle_1 = [[[16400, 1]], [[16600, 1]]]
pos_strangle_2 = [[[16200, 1]], [[16800, 1]]]
psg1 = OptionProfit(pos_strangle_1)
psg2 = OptionProfit(pos_strangle_2)

#show plot
plt.plot(ST, pb1, 'b',  [ST[0], ST[-1]], [0,0], 'k--')
plt.plot(ST, pb2, 'c', [ST[0], ST[-1]], [0,0], 'k--')
plt.plot(ST, psd, 'r', [ST[0], ST[-1]], [0,0], 'k--')
plt.plot(ST, psg1, 'g', [ST[0], ST[-1]], [0,0], 'k--')
plt.plot(ST, psg2, 'm', [ST[0], ST[-1]], [0,0], 'k--')
plt.show()

#2

#find upper and lower profit line
price = 16100
for i in range(8):
    pos_i = [[[price, 1]],[[price, -1]]] 
    pi = OptionProfit(pos_i)
    print(max(pi)) # upper profit line is 16800 and lower profit line is 16200
    price += 100
    
print(max(OptionProfit([[[16800, 1]], [[16800, -1]]]))-max(OptionProfit([[[16200, 1]], [[16200, -1]]]))) # max profit

pos_max = [[[16800, 1]], [[16800, -1]]]
pos_min = [[[16200, 1]], [[16200, -1]]]
pmax = OptionProfit(pos_max)
pmin = OptionProfit(pos_min)

plt.plot(ST, pmax, 'r', [ST[0], ST[-1]], [0,0], 'y--')   
plt.plot(ST, pmin, 'k', [ST[0], ST[-1]], [0,0], 'y--') 
plt.show()

#3

#call only x2
pos_call_1 = [[[16100, -1]], []]
pos_call_2 = [[[16300, -1]], []]
pc1 = OptionProfit(pos_call_1)
pc2 = OptionProfit(pos_call_2)

#put only x2
pos_put_1 = [[], [[16800, 1]]]
pos_put_2 = [[], [[16600, 1]]]
pp1 = OptionProfit(pos_put_1)
pp2 = OptionProfit(pos_put_2)

#spread x2
pos_spread_1 = [[[16800, 1], [16100, -1]], []]
pos_spread_2 = [[[16600, 1], [16100, -1]], []]
ps1 = OptionProfit(pos_spread_1)
ps2 = OptionProfit(pos_spread_2)

#show plot
plt.plot(ST, pc1, 'b', [ST[0], ST[-1]], [0,0], 'g--')
plt.plot(ST, pc2, 'c', [ST[0], ST[-1]], [0,0], 'g--')
plt.plot(ST, pp1, 'm', [ST[0], ST[-1]], [0,0], 'g--')
plt.plot(ST, pp2, 'r', [ST[0], ST[-1]], [0,0], 'g--')
plt.plot(ST, ps1, 'k', [ST[0], ST[-1]], [0,0], 'g--')
plt.plot(ST, ps2, 'y', [ST[0], ST[-1]], [0,0], 'g--')
plt.show()