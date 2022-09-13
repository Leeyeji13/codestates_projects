import csv
import random
import pickle
from flask import Blueprint, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

main_bp = Blueprint('main', __name__)

model = pickle.load(open('model.pkl', 'rb'))

@main_bp.route('/', methods=['GET'])
def index():
    with open("./flask_app/track_df.csv", 'r', encoding = 'utf-8') as f:
      reader = csv.reader(f)
      next(reader)
      tracklist = list(reader)
      tracklist_50 = random.sample(tracklist, 10)

    return render_template('index.html', track_list = tracklist_50), 200


@main_bp.route('/post', methods=['GET','POST'])
def tmp():
    track_id = request.form.get('like') ##track_id 빼왔어

    #추천해주자
    track_df = pd.read_csv("./flask_app/track_df.csv", encoding = 'utf-8')
    count_vector = CountVectorizer(ngram_range=(1,3))
    c_vector_genres = count_vector.fit_transform(track_df['artist_genres'].values.astype('U'))
    genre_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1] 

    def get_recommend_track_list(df, track_name, top = 10):
        sim_ind=[]
        #특정 트랙과 비슷한 트랙을 추천하기 때문에 특정 트랙 정보를 뽑아낸다.
        target_track_index = df[df['track_name'] == track_name].index.values
        #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
        sim_index = genre_c_sim[target_track_index, :top].reshape(-1)
        #본인을 제외
        sim_index = sim_index[sim_index!=target_track_index]
        sim_index = sim_index[0]

        #dataframe으로 만들고 return
        result =  df.iloc[sim_index[:]]
        result = result.drop_duplicates(['artist_name','track_name'])
        result = result[:top].sort_values('like',ascending = False)
        return result
    #추천
    trackname = track_df.loc[track_df['track_id'] == track_id, ['track_name']]
    df = get_recommend_track_list(track_df, track_name = trackname)
    value = df.values.tolist()

    #예측모델
    data = pd.read_csv("./flask_app/df.csv", encoding = 'utf-8')
    row = data[data['track_id'] == track_id]
    row = row.drop(['track_id', 'like'], axis = 1)
    pred = model.predict(row)
    return render_template('post.html', pred = pred, value = value)