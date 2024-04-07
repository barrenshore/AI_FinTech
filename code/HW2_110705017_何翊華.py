R = 0.0194 #年利率
r = R/12 #月利率
n = 12*20 #總期數
delay = 12*2 #寬限期

def calculate(p, n, delay, rate): #input 總餘額  總期數 寬限期 月利率
    count = 1 #計算月份
    interest = 0 #計算利息總合
    
    #寬限期
    for i in range(delay): 
        print("月份 " + str(count) + " 應還本金 0 應付利息 " + str(p*r) + " 應付本息 "+ str(p*r))
        interest += p*r
        count += 1
        
    #寬限期後
    c = p*(r/(1-(1+r)**(delay-n))) #平均攤還公式
    prior = p #上月餘額
    
    for i in range(n-delay): 
        p = p*(1+r)-c
        print("月份 " + str(count) + " 應還本金 " + str(prior-p) + " 應付利息 " + str(c - prior + p) + " 應付本息 "+ str(c))
        interest += c - prior + p
        prior = p
        count+=1
    
    print("利息總合: " + str(interest))
        
        
calculate(20000000, n, delay, r)