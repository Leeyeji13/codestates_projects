#마지막 주(20.09.16.-20.09.22.) top 10 아이템 추천하기
import pandas as pd
import numpy as np

transactions = pd.read_csv("./transactions_train.csv", sep = ",")
submission = pd.read_csv("./sample_submission.csv", sep = ",")

transactions = transactions.dropna()
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
transactions = transactions.loc[transactions.t_dat >= pd.to_datetime('2020-09-16')]
top10 = ' '.join(transactions.article_id.value_counts().index.astype('str')[:10])

print(top10)