import bisect
T=int(input())
for t in range(T):
  N=int(input())
  buy=[]
  sell=[]
  stockprice=-1
  for n in range(N):
    str=input()
    strlist=str.split()
    share=int(strlist[1])
    price=int(strlist[-1])
    if strlist[0]=="buy":
      while len(sell)>0:
        order=sell[0]
        if order[0]>price:
          break
        dealamount=min(order[1],share)
        stockprice=order[0]
        order[1]-=dealamount
        share-=dealamount
        if order[1]==0:
          del sell[0]
        if share==0:
          break
      if share>0:
        i=0
        while(i<len(buy) and price<buy[i][0]):
          i+=1
        if(i<len(buy) and price==buy[i][0]):
          buy[i][1]+=share
        else:
          buy.insert(i,[price,share])


    elif strlist[0]=="sell":
      while len(buy)>0:
        order=buy[0]
        if order[0]<price:
          break
        dealamount=min(order[1],share)
        stockprice=price
        order[1]-=dealamount
        share-=dealamount
        if order[1]==0:
          del buy[0]
        if share==0:
          break
      if share > 0:
        # 創建新的賣出訂單
        new_order = [price, share]

        # 使用 bisect.insort 插入新訂單，同時保持升序排序
        bisect.insort(sell, new_order)



    if not sell:
      print('-',end=' ')
    else:
      print(sell[0][0],end=' ')
    if not buy:
      print('-',end=' ')
    else:
      print(buy[0][0],end=' ')
    if stockprice==-1:
      print('-')
    else:
      print(stockprice)
            
            
            
            
            
            
            
            