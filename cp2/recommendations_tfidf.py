#상품에 대한 설명을 자연어 처리하여 이전에 구매한 아이템과 유사한 상품 추천하기(tf-idf)
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
from sklearn.metrics.pairwise import cosine_similarity
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


#상품 설명 : 맥락을 고려할 필요가 없고, 벡터 공간 내 단어 할당을 임의적으로 줘도 상관 X -> 임베딩이 아닌 텍스트 벡터화로 진행
#머신러닝 모델에서 텍스트를 분석하기 위해서는 벡터화(텍스트를 컴퓨터가 계산할 수 있도록 수치정보로 변환하는 과정)하는 과정이 필요
#다른 문서에 등장하지 않는 단어, 즉 특정 문서에만 등장하는 단어에 가중치를 두는 방법-> TF-IDF
#상품 설명에 대해 벡터화
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
cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf_article)

cos_sim_list = list(cos_similarity_tfidf) 
top_10 = sorted(range(len(cos_sim_list)), key=lambda i: cos_sim_list[i], reverse = True)[:10]
tf_list_scores = [cos_sim_list[i][0][0] for i in top_10]
pd.set_option('display.max_columns', None)
print(recommendation_product(top_10, df, tf_list_scores))