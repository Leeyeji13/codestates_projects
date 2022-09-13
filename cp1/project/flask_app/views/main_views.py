import csv
import random
import pickle
from flask import Blueprint, render_template, request, make_response
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html'), 200

@main_bp.route('/getcookie', methods=['GET', 'POST'])
def getcookie():
    return request.cookies.get('userID') #class str 이다!

@main_bp.route('/post1', methods=['GET','POST'])
def tmp1():
    if request.method == 'POST':
        area = request.form['areaCode']
        resp = make_response(render_template('tag.html'), 200)
        resp.set_cookie('areaName', area)
        return resp
#return render_template('tag.html'), 200

@main_bp.route('/post2', methods=['GET','POST'])
def tmp2():
    name = request.cookies.get('areaName') #class str 이다!
    all_data = request.form
    if request.method == 'POST':
        tag_list = all_data.getlist('tag')
        df = pd.read_csv("data.csv", index_col = 0)
        cv = CountVectorizer()
        cat = cv.fit_transform(df.cat1)
        cat1=pd.DataFrame(
            cat.toarray(),
            columns=list(sorted(cv.vocabulary_.keys(),key=lambda x : cv.vocabulary_[x]))
            )
        cat1 = cat1.T

        nbrs = NearestNeighbors(n_neighbors = 4).fit(cat1)

        if len(tag_list) == 1:
            if name == '전국':
                a = cat1.loc[tag_list[0]]
                distances, indexes = nbrs.kneighbors([a])
                recommendations = df.loc[indexes[0], ["title", "firstimage", "firstimage2"]]
                recommendations["distance"] = distances[0]
                return render_template('post.html', value=recommendations), 200

            else:
                df = df.loc[df['addr_area'] == name, :]
                a = cat1.loc[tag_list[0]]
                distances, indexes = nbrs.kneighbors([a])
                recommendations = df.iloc[indexes[0]]
                recommendations["distance"] = distances[0]
                return render_template('post.html', value=recommendations), 200

        if len(tag_list) == 2:
            a = cat1.loc[tag_list[0]]
            b = cat1.loc[tag_list[1]]
            distances, indexes = nbrs.kneighbors([a+b])
            if name == '전국':
                recommendations = df.loc[indexes[0], ["title", "firstimage", "firstimage2"]]
                recommendations["distance"] = distances[0]
                return render_template('post.html', value=recommendations), 200

            else:
                df = df.loc[df['addr_area'] == name, :]
                distances, indexes = nbrs.kneighbors([a+b])
                recommendations = df.iloc[indexes[0]]
                recommendations["distance"] = distances[0]
                return render_template('post.html', value=recommendations), 200

        if len(tag_list) == 3:
            a = cat1.loc[tag_list[0]]
            b = cat1.loc[tag_list[1]]
            c = cat1.loc[tag_list[2]]
            distances, indexes = nbrs.kneighbors([a+b+c])
            if name == '전국':
                recommendations = df.loc[indexes[0], ["title", "firstimage", "firstimage2"]]
                recommendations["distance"] = distances[0]
                return render_template('post.html', value=recommendations), 200

            else:
                df = df.loc[df['addr_area'] == name, :]
                distances, indexes = nbrs.kneighbors([a+b+c])
                recommendations = df.iloc[indexes[0]]
                recommendations["distance"] = distances[0]
                return render_template('post.html', value=recommendations), 200