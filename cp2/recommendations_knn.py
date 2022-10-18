#상품에 대한 설명을 자연어 처리하여 이전에 구매한 아이템과 유사한 상품 추천하기(KNN)
import pandas as pd
import numpy as np

from warnings import filterwarnings
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

articles = pd.read_csv("./articles.csv", sep = ",")
customers = pd.read_csv("./customers.csv", sep = ",")
transactions = pd.read_csv("./transactions_train.csv", sep = ",")

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

#(제거) 불용어, 특수문자, 길이 2 이하 토큰
def del_txt(token):
    return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2

#(변경) 정규식, 토큰화, 소문자, 표제어
def change_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "",text)
  text = re.sub("(\\d|\\W)+"," ",text) ##숫자|알파벳이나 숫자를 제외한 문자
  text = text.replace("nbsp", "") ##non-breaking-space(줄바꿈방지공백 : 한 줄의 마지막에 있는 단어는 잘리지 않고 전체가 아래로 내려가는 경우)
  clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if del_txt(word)]
  clean_text2 = [word for word in clean_text if del_txt(word)]
  return " ".join(clean_text2)

articles = articles.dropna()
transactions = transactions.dropna()
articles["text"] = articles["prod_name"].map(str) + " " + articles["product_type_name"] + " " + articles["product_group_name"] + " " + articles['graphical_appearance_name'] + " " + articles['colour_group_name'] + " " + articles['perceived_colour_value_name'] + " " + articles["perceived_colour_master_name"] + " " + articles["department_name"] + " " + articles['index_name'] +" " + articles['index_group_name'] + " " + articles['section_name'] + " " + articles['garment_group_name'] + " " + articles['detail_desc']
df = articles[['article_id', 'text']] #자연어 처리(제품에 대한 설명 자연어 처리)를 위해 제품 번호와 text만
df['text'] = df['text'].apply(change_txt)

tfidf_vectorizer = TfidfVectorizer()
tfidf_article = tfidf_vectorizer.fit_transform((df['text']))

transactions = transactions.sort_values(by='customer_id')

merged_df = df.merge(transactions, how = 'inner', on = ['article_id'])
merged_df = merged_df.groupby('customer_id', sort=False)['text'].apply(' '.join).reset_index()

#random으로 설정해주기
random_user = "8536c0c8b77f15197e75eb25aaf11663732b632f6e2abcadd1907e9f372f108f" #customer_id
index = np.where(merged_df['customer_id'] == random_user)[0][0]
q = merged_df.iloc[[index]]

def recommendation_product(top, df, scores):
  recommendation = pd.DataFrame(columns = ['customer_id', 'article_id', 'score', 'text'])
  count = 0
  for i in top:
      recommendation.at[count, 'customer_id'] = random_user
      recommendation.at[count, 'article_id'] = df['article_id'][i]
      recommendation.at[count, 'score'] =  scores[count]
      recommendation.at[count, 'text'] = df['text'][i]  
      count += 1
  return recommendation

user_tfidf = tfidf_vectorizer.transform(q['text'])

KNN = NearestNeighbors(n_neighbors = 11) #가장 가까운 거리에 있는 11개의 데이터를 찾음
KNN.fit(tfidf_article)
NNs = KNN.kneighbors(user_tfidf, return_distance=True) #user_tfidf 지점의 11개의 이웃들을 찾아줌

top_10 = NNs[1][0][1:]
index_score = NNs[0][0][1:]
pd.set_option('display.max_columns', None)
print(recommendation_product(top_10, df, index_score))