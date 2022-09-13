##pip freeze > requirements.txt
##pip install -r requirements.txt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

#1.spotipy api 연결 
client_credentials_manager = SpotifyClientCredentials(client_id='5cbabba598354dcb9ed045ac470f346c', client_secret='45e65ad53775447c89b7504119f09f41')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#2.데이터 가져오기
##2021년 트랙 데이터(트랙의 아티스트, 제목, 아이디, 인기 등의 정보) (리스트를 데이터프레임으로 변환)
###spotipy의 search 함수를 사용해 2021년 트랙만 검색
artist_name =[]
track_name = []
track_popularity =[]
artist_id =[]
track_id =[]
for i in range(0,1000,50):
    track_results = sp.search(q='year:2021', type='track', limit=50, offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        artist_id.append(t['artists'][0]['id'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        track_popularity.append(t['popularity'])

track_df = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'track_popularity' : track_popularity, 'artist_id' : artist_id})

##아티스트 데이터(트랙 데이터(아티스트의 이름과 아이디)에 추가(인기, 장르, 팔로워수)해주기)
###spotipy의 artist 함수로 아티스트의 url, 팔로워수, 장르, 이미지, 이름, 타입, 인기 등의 정보 받기
artist_popularity = []
artist_genres = []
artist_followers =[]
for a_id in track_df.artist_id:
    artist = sp.artist(a_id)
    artist_popularity.append(artist['popularity'])
    artist_genres.append(artist['genres'])
    artist_followers.append(artist['followers']['total'])

track_df = track_df.assign(artist_popularity=artist_popularity, artist_genres=artist_genres, artist_followers=artist_followers)

#오디오 데이터
##스포티파이는 트랙에 대해 17개의 Audio features를 제공
track_features = []
for t_id in track_df['track_id']:
    af = sp.audio_features(t_id)
    track_features.append(af)
tf_df = pd.DataFrame(columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'url', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'])
for item in track_features:
    for feat in item:
        tf_df = pd.concat([tf_df, pd.DataFrame.from_records([feat])], ignore_index=True)


#3.전처리 (track_df = 트랙(웹에서 사용-정보 접근), tf_df = 오디오, df = 오디오 특성 + 타겟(예측모델에서 사용))
##필요없는 열 제거
cols_to_drop2 = ['key','mode','type', 'url','track_href','analysis_url', 'uri']
tf_df = tf_df.drop(columns=cols_to_drop2)

###데이터 타입 변경
track_df['artist_name'] = track_df['artist_name'].astype("string")
track_df['track_name'] = track_df['track_name'].astype("string")
track_df['track_id'] = track_df['track_id'].astype("string")
track_df['artist_id'] = track_df['artist_id'].astype("string")
tf_df['duration_ms'] = pd.to_numeric(tf_df['duration_ms'])
tf_df['instrumentalness'] = pd.to_numeric(tf_df['instrumentalness'])
tf_df['time_signature'] = tf_df['time_signature'].astype("category")

###트랙의 인기가 불공정하다고 판단(아티스트 인기가 많을수록 트랙의 인기가 많아질수밖에 없다고 판단)
m = track_df['artist_popularity'].quantile(0.9)
C = track_df['track_popularity'].mean()

def weighted_popularity(x, m = m, C = C):
  v = x['artist_popularity']
  R = x['track_popularity']
  return ( v / (v + m ) * R ) + ( m / ( m + v ) * C )

track_df['like'] = track_df.apply(weighted_popularity, axis = 1)

df = pd.merge(track_df, tf_df, left_on = "track_id", right_on = "id")
cols_to_drop3 = ['track_name', 'artist_name', 'track_popularity', 'artist_id', 'id', 'artist_popularity', 'artist_genres', 'artist_followers']
df = df.drop(columns = cols_to_drop3)

###데이터 핸들링
track_df['artist_genres'] = track_df['artist_genres'].apply(lambda x: ', '.join(dict.fromkeys(x).keys()))


##4.콘텐츠 기반 필터링 추천
###비슷한 장르를 추천해주는 것이기 때문에 장르 활용 예정
count_vector = CountVectorizer(ngram_range=(1,3)) #객체 만들어주기
c_vector_genres = count_vector.fit_transform(track_df['artist_genres']) #변환시켜주기
genre_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1] #코사인 유사도를 구함과 동시에 argsort로 유사도가 가장 높은 인덱스를 가장 위쪽으로 정렬

def get_recommend_track_list(df, track_name, top = 5):
  #특정 트랙과 비슷한 트랙을 추천하기 때문에 특정 트랙 정보를 뽑아낸다.
  target_track_index = df[df['track_name'] == track_name].index.values
  #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
  sim_index = genre_c_sim[target_track_index, :top].reshape(-1)
  #본인을 제외
  sim_index = sim_index[sim_index != target_track_index]
  #dataframe으로 만들고 artist popularity로 정렬한 뒤 return
  result = df.iloc[sim_index].sort_values('like',ascending = False)[:10]
  return result

###ml과 웹을 위해 csv 파일 생성
track_df.to_csv('C:\\Users\\santa\\OneDrive\\바탕 화면\\P3\\flask_app\\track_df.csv', index = False)
df.to_csv('C:\\Users\\santa\\OneDrive\\바탕 화면\\P3\\flask_app\\df.csv', index = False)

# ###mongodb에 적재하기 위해 데이터프레임에서 json으로 변환 필수
# ####key와 value로 접근하기 위한 dict로 변환하려면 개별 map에 대해 json.loads로 해주어야 한다
# track_json = json.loads(track_df.to_json())
# df_json = json.loads(df.to_json())

# #4.데이터 적재 mongodb(collection-테이블)
# HOST = 'cluster0.kwsuy.mongodb.net'
# USER = 'project_3rd'
# PASSWORD = 'p3p3'
# DATABASE_NAME = 'Section3Project'
# COLLECTION_NAME1 = 'track_data'
# COLLECTION_NAME2 = 'data'

# MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

# client = MongoClient(MONGO_URI)
# database = client[DATABASE_NAME]
# collection1 = database[COLLECTION_NAME1]
# collection2 = database[COLLECTION_NAME2]

# collection1.insert_one(track_json)
# collection2.insert_one(df_json)

# client.close()
