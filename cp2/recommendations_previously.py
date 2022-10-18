#고객별 이전에 자주 구매한 아이템 추천하기
import pandas as pd
import numpy as np

customers = pd.read_csv("./customers.csv", sep = ",")
transactions = pd.read_csv("./transactions_train.csv", sep = ",")

transactions = transactions.dropna()

transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

purchase_dict = {}
#ex) '8536c0c8b77f15197e75eb25aaf11663732b632f6e2abcadd1907e9f372f108f': {562245001: 1, 562245059: 1}

for i, x in enumerate(zip(transactions['customer_id'], transactions['article_id'])):
    customer_id, article_id = x
    if customer_id not in purchase_dict:
        purchase_dict[customer_id] = {}
    
    if article_id not in purchase_dict[customer_id]:
        purchase_dict[customer_id][article_id] = 0
    
    purchase_dict[customer_id][article_id] += 1

random = customers[['customer_id']]
recommendation_list = []

article_top_10_list = list((transactions['article_id'].value_counts()).index)[:10]
#[706016001, 706016002, 372860001, 610776002, 759871002, 464297007, 372860002, 610776001, 399223001, 706016003]

for i in range(len(article_top_10_list)):
    article_top_10_list[i] = str(article_top_10_list[i])

recommendation = ' '.join(article_top_10_list)

for i, cust_id in enumerate(customers['customer_id'].values.reshape((-1,))):
    if cust_id in purchase_dict:
        l = sorted((purchase_dict[cust_id]).items(), key = lambda x: x[1], reverse = True)
        l = [y[0] for y in l] #제품번호만 뽑기
        for i in range(len(l)):
            l[i] = str(l[i])
        if len(l)>10:
            s = ' '.join(l[:10])
        else:
            s = ' '.join(l + article_top_10_list[:(10 - len(l))])
    else:
        s = recommendation
    recommendation_list.append(s)

random['recommendations'] = recommendation_list

print(random.head(5))