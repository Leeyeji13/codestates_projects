#4.데이터 적재 sqlite
import os
import sqlite3
import csv

DB_FILENAME = 'DATA.db'
DB_FILEPATH = os.path.join(os.getcwd(), DB_FILENAME)

conn = sqlite3.connect(DB_FILENAME)

cur = conn.cursor()

##track 테이블 생성
cur.execute("DROP TABLE IF EXISTS track;")

cur.execute("""CREATE TABLE track(
				artist_name VARCHAR(128),
                track_name VARCHAR(128),
                track_id VARCHAR(128) NOT NULL PRIMARY KEY,
                track_popularity VARCHAR(128),
                artist_id VARCHAR(128),
                artist_popularity VARCHAR(128),
                artist_genres VARCHAR(128),
                artist_followers VARCHAR(128),
                like VARCHAR(128));
			""")

##track 테이블 데이터 입력
file1 = open('./flask_app/track_df.csv', 'r', encoding='utf-8')
row1 = csv.reader(file1)
next(row1)

for index, features in enumerate(row1):
    row1 = [index]
    for feature in features:
        row1.append(feature)
    cur.execute("INSERT INTO track VALUES(?,?,?,?,?,?,?,?,?)", tuple(row1[1:]))
conn.commit()

##audio feature 테이블생성
cur.execute("DROP TABLE IF EXISTS audio;")

cur.execute("""CREATE TABLE audio(
				track_id VARCHAR(128) NOT NULL PRIMARY KEY,
                like VARCHAR(128),
                danceability VARCHAR(128),
                energy VARCHAR(128),
                loudness VARCHAR(128),
                speechiness VARCHAR(128),
                acousticness VARCHAR(128),
                instrumentalness VARCHAR(128),
                liveness VARCHAR(128),
                valence VARCHAR(128),
                tempo VARCHAR(128),
                duration_ms VARCHAR(128),
                time_signature VARCHAR(128),
                FOREIGN KEY(track_id) REFERENCES track(track_id));
			""")
            
##track 테이블 데이터 입력
file2 = open('./flask_app/df.csv', 'r', encoding = 'utf-8')
row2 = csv.reader(file2)
next(row2)

for index, features in enumerate(row2):
    row2 = [index]
    for feature in features:
        row2.append(feature)
    cur.execute("INSERT INTO audio VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(row2[1:]))
conn.commit()

conn.close()
