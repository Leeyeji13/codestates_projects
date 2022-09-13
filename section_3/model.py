
#다중 선형 회귀 모델(음악 특성으로 좋아하는지를 예측하는 모델) 제작
#랜덤 플레이리스트를 (트랙 리스트 중 랜덤 50개를 뽑기) 모델에 넣은 결과로 노래를 추천할 수 있지 않을까 하는 생각에서 시작
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('./flask_app/df.csv', encoding = 'utf-8')
df.drop(['track_id'], axis = 1, inplace = True)

target = ['like']
X = df.drop(columns = target)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, test_size = 0.20, random_state = 2)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl','wb'))