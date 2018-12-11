---
title: Connecting to Database
notebook: Connect_to_database.ipynb
nav_include: 1
---

SQLite Python Tutorial

http://www.sqlitetutorial.net/sqlite-python/

https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html



```python
import sqlite3
import pandas as pd
import spotipy
import spotipy.oauth2 as oauth2
import pickle
import pandas.io.sql as psql
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
```




```python
# connect to the database
conn = sqlite3.connect("spotifyDB.db")
c = conn.cursor()  # get a cursor
```




```python
# view all tables in database
table_names = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in table_names:
    print(name[0])
```


    sqlite_sequence
    tracks
    playlists
    Million_songs
    combined_table
    track_song_combine
    track_features
    BigTables
    BigTable




```python
# set up authorization token
credentials = oauth2.SpotifyClientCredentials(
        client_id='153369a05314402294db1a574caaff2a',
        client_secret='c6fff0923a0c44c5851fc4415038e8fa')

token = credentials.get_access_token()
sp = spotipy.Spotify(auth=token)
```




```python
c.execute("SELECT track FROM tracks")  # execute a simple SQL select query
jobs = c.fetchall()  # get all the results from the above query
```




```python
with open('tracks_10000_rand.pkl', 'rb') as f:
    X = pickle.load(f)
```




```python
# returns the next n elements from the iterator
# used because Spotify limits how many items you can group into each of its API calls -- in this case to 100
import itertools
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
```




```python
# get the audio features of each of the tracks fetched above
track_features = {}

# can only take 100 songs at a time
for group in grouper(100, X[:1000]):
    track_features = sp.audio_features(tracks=group)
#     for item in res:
#         track_features[item['id']] = item
# track_features = spotify.audio_features(tracks=X[:100])
# track_features
# creates lists of dicts instead of list of dicts
# track_features = []

track_features
```





    [{'danceability': 0.839,
      'energy': 0.791,
      'key': 8,
      'loudness': -7.771,
      'mode': 1,
      'speechiness': 0.116,
      'acousticness': 0.0531,
      'instrumentalness': 0.000485,
      'liveness': 0.201,
      'valence': 0.858,
      'tempo': 129.244,
      'type': 'audio_features',
      'id': '1YGa5zwwbzA9lFGPB3HcLt',
      'uri': 'spotify:track:1YGa5zwwbzA9lFGPB3HcLt',
      'track_href': 'https://api.spotify.com/v1/tracks/1YGa5zwwbzA9lFGPB3HcLt',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1YGa5zwwbzA9lFGPB3HcLt',
      'duration_ms': 267333,
      'time_signature': 4},
     {'danceability': 0.728,
      'energy': 0.779,
      'key': 4,
      'loudness': -7.528,
      'mode': 0,
      'speechiness': 0.19,
      'acousticness': 0.00336,
      'instrumentalness': 1.18e-06,
      'liveness': 0.325,
      'valence': 0.852,
      'tempo': 174.046,
      'type': 'audio_features',
      'id': '4gOMf7ak5Ycx9BghTCSTBL',
      'uri': 'spotify:track:4gOMf7ak5Ycx9BghTCSTBL',
      'track_href': 'https://api.spotify.com/v1/tracks/4gOMf7ak5Ycx9BghTCSTBL',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4gOMf7ak5Ycx9BghTCSTBL',
      'duration_ms': 278810,
      'time_signature': 4},
     {'danceability': 0.314,
      'energy': 0.855,
      'key': 2,
      'loudness': -7.907,
      'mode': 1,
      'speechiness': 0.034,
      'acousticness': 0.447,
      'instrumentalness': 0.854,
      'liveness': 0.173,
      'valence': 0.855,
      'tempo': 104.983,
      'type': 'audio_features',
      'id': '7kl337nuuTTVcXJiQqBgwJ',
      'uri': 'spotify:track:7kl337nuuTTVcXJiQqBgwJ',
      'track_href': 'https://api.spotify.com/v1/tracks/7kl337nuuTTVcXJiQqBgwJ',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7kl337nuuTTVcXJiQqBgwJ',
      'duration_ms': 451160,
      'time_signature': 4},
     {'danceability': 0.762,
      'energy': 0.954,
      'key': 8,
      'loudness': -4.542,
      'mode': 1,
      'speechiness': 0.121,
      'acousticness': 0.22,
      'instrumentalness': 1.76e-05,
      'liveness': 0.0612,
      'valence': 0.933,
      'tempo': 153.96,
      'type': 'audio_features',
      'id': '0LAfANg75hYiV1IAEP3vY6',
      'uri': 'spotify:track:0LAfANg75hYiV1IAEP3vY6',
      'track_href': 'https://api.spotify.com/v1/tracks/0LAfANg75hYiV1IAEP3vY6',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0LAfANg75hYiV1IAEP3vY6',
      'duration_ms': 271907,
      'time_signature': 4},
     {'danceability': 0.633,
      'energy': 0.834,
      'key': 11,
      'loudness': -12.959,
      'mode': 1,
      'speechiness': 0.0631,
      'acousticness': 0.422,
      'instrumentalness': 0.726,
      'liveness': 0.172,
      'valence': 0.518,
      'tempo': 130.008,
      'type': 'audio_features',
      'id': '0Hpl422q9VhpQu1RBKlnF1',
      'uri': 'spotify:track:0Hpl422q9VhpQu1RBKlnF1',
      'track_href': 'https://api.spotify.com/v1/tracks/0Hpl422q9VhpQu1RBKlnF1',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0Hpl422q9VhpQu1RBKlnF1',
      'duration_ms': 509307,
      'time_signature': 4},
     {'danceability': 0.724,
      'energy': 0.667,
      'key': 4,
      'loudness': -4.806,
      'mode': 0,
      'speechiness': 0.385,
      'acousticness': 0.126,
      'instrumentalness': 0,
      'liveness': 0.122,
      'valence': 0.161,
      'tempo': 143.988,
      'type': 'audio_features',
      'id': '4uTTsXhygWzSjUxXLHZ4HW',
      'uri': 'spotify:track:4uTTsXhygWzSjUxXLHZ4HW',
      'track_href': 'https://api.spotify.com/v1/tracks/4uTTsXhygWzSjUxXLHZ4HW',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4uTTsXhygWzSjUxXLHZ4HW',
      'duration_ms': 231824,
      'time_signature': 4},
     {'danceability': 0.617,
      'energy': 0.493,
      'key': 0,
      'loudness': -12.779,
      'mode': 1,
      'speechiness': 0.0495,
      'acousticness': 0.221,
      'instrumentalness': 0.294,
      'liveness': 0.696,
      'valence': 0.148,
      'tempo': 119.003,
      'type': 'audio_features',
      'id': '1KDnLoIEPRd4iRYzgvDBzo',
      'uri': 'spotify:track:1KDnLoIEPRd4iRYzgvDBzo',
      'track_href': 'https://api.spotify.com/v1/tracks/1KDnLoIEPRd4iRYzgvDBzo',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1KDnLoIEPRd4iRYzgvDBzo',
      'duration_ms': 259147,
      'time_signature': 4},
     {'danceability': 0.353,
      'energy': 0.304,
      'key': 4,
      'loudness': -9.142,
      'mode': 0,
      'speechiness': 0.0327,
      'acousticness': 0.435,
      'instrumentalness': 1.54e-06,
      'liveness': 0.802,
      'valence': 0.233,
      'tempo': 138.494,
      'type': 'audio_features',
      'id': '0j9rNb4IHxLgKdLGZ1sd1I',
      'uri': 'spotify:track:0j9rNb4IHxLgKdLGZ1sd1I',
      'track_href': 'https://api.spotify.com/v1/tracks/0j9rNb4IHxLgKdLGZ1sd1I',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0j9rNb4IHxLgKdLGZ1sd1I',
      'duration_ms': 407814,
      'time_signature': 4},
     {'danceability': 0.633,
      'energy': 0.286,
      'key': 2,
      'loudness': -9.703,
      'mode': 0,
      'speechiness': 0.027,
      'acousticness': 0.221,
      'instrumentalness': 0.00415,
      'liveness': 0.0879,
      'valence': 0.196,
      'tempo': 129.915,
      'type': 'audio_features',
      'id': '1Dfst5fQZYoW8QBfo4mUmn',
      'uri': 'spotify:track:1Dfst5fQZYoW8QBfo4mUmn',
      'track_href': 'https://api.spotify.com/v1/tracks/1Dfst5fQZYoW8QBfo4mUmn',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1Dfst5fQZYoW8QBfo4mUmn',
      'duration_ms': 205267,
      'time_signature': 4},
     {'danceability': 0.569,
      'energy': 0.789,
      'key': 11,
      'loudness': -4.607,
      'mode': 1,
      'speechiness': 0.123,
      'acousticness': 0.564,
      'instrumentalness': 0,
      'liveness': 0.294,
      'valence': 0.603,
      'tempo': 160.014,
      'type': 'audio_features',
      'id': '6ZbiaHwI9x7CIxYGOEmXxd',
      'uri': 'spotify:track:6ZbiaHwI9x7CIxYGOEmXxd',
      'track_href': 'https://api.spotify.com/v1/tracks/6ZbiaHwI9x7CIxYGOEmXxd',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6ZbiaHwI9x7CIxYGOEmXxd',
      'duration_ms': 198012,
      'time_signature': 4},
     {'danceability': 0.726,
      'energy': 0.718,
      'key': 7,
      'loudness': -5.192,
      'mode': 0,
      'speechiness': 0.051,
      'acousticness': 0.407,
      'instrumentalness': 0.000266,
      'liveness': 0.12,
      'valence': 0.67,
      'tempo': 123.981,
      'type': 'audio_features',
      'id': '3AUz2xcMnn1CDLDtFRCPeV',
      'uri': 'spotify:track:3AUz2xcMnn1CDLDtFRCPeV',
      'track_href': 'https://api.spotify.com/v1/tracks/3AUz2xcMnn1CDLDtFRCPeV',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3AUz2xcMnn1CDLDtFRCPeV',
      'duration_ms': 214520,
      'time_signature': 4},
     {'danceability': 0.524,
      'energy': 0.904,
      'key': 8,
      'loudness': -2.071,
      'mode': 1,
      'speechiness': 0.398,
      'acousticness': 0.0533,
      'instrumentalness': 0,
      'liveness': 0.776,
      'valence': 0.655,
      'tempo': 161.188,
      'type': 'audio_features',
      'id': '7qtAgn9mwxygsPOsUDVRRt',
      'uri': 'spotify:track:7qtAgn9mwxygsPOsUDVRRt',
      'track_href': 'https://api.spotify.com/v1/tracks/7qtAgn9mwxygsPOsUDVRRt',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7qtAgn9mwxygsPOsUDVRRt',
      'duration_ms': 254040,
      'time_signature': 4},
     {'danceability': 0.808,
      'energy': 0.404,
      'key': 1,
      'loudness': -10.124,
      'mode': 0,
      'speechiness': 0.0427,
      'acousticness': 0.95,
      'instrumentalness': 0.79,
      'liveness': 0.124,
      'valence': 0.84,
      'tempo': 98.023,
      'type': 'audio_features',
      'id': '6X6ctfqSkAaaUWNAEt3J3L',
      'uri': 'spotify:track:6X6ctfqSkAaaUWNAEt3J3L',
      'track_href': 'https://api.spotify.com/v1/tracks/6X6ctfqSkAaaUWNAEt3J3L',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6X6ctfqSkAaaUWNAEt3J3L',
      'duration_ms': 161307,
      'time_signature': 4},
     {'danceability': 0.665,
      'energy': 0.426,
      'key': 2,
      'loudness': -11.557,
      'mode': 1,
      'speechiness': 0.0308,
      'acousticness': 0.562,
      'instrumentalness': 0.902,
      'liveness': 0.1,
      'valence': 0.684,
      'tempo': 98.381,
      'type': 'audio_features',
      'id': '0MzO0c9Lr1d4mTUQtGhSJX',
      'uri': 'spotify:track:0MzO0c9Lr1d4mTUQtGhSJX',
      'track_href': 'https://api.spotify.com/v1/tracks/0MzO0c9Lr1d4mTUQtGhSJX',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0MzO0c9Lr1d4mTUQtGhSJX',
      'duration_ms': 353973,
      'time_signature': 4},
     {'danceability': 0.413,
      'energy': 0.959,
      'key': 6,
      'loudness': -4.007,
      'mode': 0,
      'speechiness': 0.158,
      'acousticness': 3.69e-05,
      'instrumentalness': 0.0287,
      'liveness': 0.464,
      'valence': 0.181,
      'tempo': 114.02,
      'type': 'audio_features',
      'id': '1FuJsPUcEyAELOp64pc7xw',
      'uri': 'spotify:track:1FuJsPUcEyAELOp64pc7xw',
      'track_href': 'https://api.spotify.com/v1/tracks/1FuJsPUcEyAELOp64pc7xw',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1FuJsPUcEyAELOp64pc7xw',
      'duration_ms': 372987,
      'time_signature': 3},
     {'danceability': 0.601,
      'energy': 0.496,
      'key': 5,
      'loudness': -7.622,
      'mode': 0,
      'speechiness': 0.23,
      'acousticness': 0.671,
      'instrumentalness': 4.28e-06,
      'liveness': 0.132,
      'valence': 0.292,
      'tempo': 177.902,
      'type': 'audio_features',
      'id': '1yka5XpwTBV951mp2OVYcn',
      'uri': 'spotify:track:1yka5XpwTBV951mp2OVYcn',
      'track_href': 'https://api.spotify.com/v1/tracks/1yka5XpwTBV951mp2OVYcn',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1yka5XpwTBV951mp2OVYcn',
      'duration_ms': 221359,
      'time_signature': 4},
     {'danceability': 0.124,
      'energy': 0.579,
      'key': 8,
      'loudness': -10.067,
      'mode': 1,
      'speechiness': 0.0464,
      'acousticness': 0.403,
      'instrumentalness': 0.888,
      'liveness': 0.226,
      'valence': 0.119,
      'tempo': 176.431,
      'type': 'audio_features',
      'id': '4RBd03PBwN3LNr0er6fkxd',
      'uri': 'spotify:track:4RBd03PBwN3LNr0er6fkxd',
      'track_href': 'https://api.spotify.com/v1/tracks/4RBd03PBwN3LNr0er6fkxd',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4RBd03PBwN3LNr0er6fkxd',
      'duration_ms': 240640,
      'time_signature': 3},
     {'danceability': 0.793,
      'energy': 0.412,
      'key': 3,
      'loudness': -10.517,
      'mode': 1,
      'speechiness': 0.21,
      'acousticness': 0.248,
      'instrumentalness': 0,
      'liveness': 0.537,
      'valence': 0.407,
      'tempo': 77.729,
      'type': 'audio_features',
      'id': '6GygUjupLLKX273CNzZ4kQ',
      'uri': 'spotify:track:6GygUjupLLKX273CNzZ4kQ',
      'track_href': 'https://api.spotify.com/v1/tracks/6GygUjupLLKX273CNzZ4kQ',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6GygUjupLLKX273CNzZ4kQ',
      'duration_ms': 80309,
      'time_signature': 4},
     {'danceability': 0.136,
      'energy': 0.217,
      'key': 7,
      'loudness': -17.341,
      'mode': 0,
      'speechiness': 0.0622,
      'acousticness': 0.96,
      'instrumentalness': 0.000174,
      'liveness': 0.963,
      'valence': 0.178,
      'tempo': 66.749,
      'type': 'audio_features',
      'id': '060oQAg8NRV2gbBTKYIRPA',
      'uri': 'spotify:track:060oQAg8NRV2gbBTKYIRPA',
      'track_href': 'https://api.spotify.com/v1/tracks/060oQAg8NRV2gbBTKYIRPA',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/060oQAg8NRV2gbBTKYIRPA',
      'duration_ms': 244507,
      'time_signature': 3},
     {'danceability': 0.387,
      'energy': 0.977,
      'key': 8,
      'loudness': -3.242,
      'mode': 1,
      'speechiness': 0.136,
      'acousticness': 0.00236,
      'instrumentalness': 0.00661,
      'liveness': 0.35,
      'valence': 0.387,
      'tempo': 139.905,
      'type': 'audio_features',
      'id': '5SzEdjMBb17oURYUXF6iGm',
      'uri': 'spotify:track:5SzEdjMBb17oURYUXF6iGm',
      'track_href': 'https://api.spotify.com/v1/tracks/5SzEdjMBb17oURYUXF6iGm',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5SzEdjMBb17oURYUXF6iGm',
      'duration_ms': 194093,
      'time_signature': 4},
     {'danceability': 0.675,
      'energy': 0.389,
      'key': 6,
      'loudness': -11.595,
      'mode': 0,
      'speechiness': 0.0356,
      'acousticness': 0.864,
      'instrumentalness': 0.000419,
      'liveness': 0.16,
      'valence': 0.415,
      'tempo': 115.923,
      'type': 'audio_features',
      'id': '6F2zt7QZkdQKKBul5ATSMD',
      'uri': 'spotify:track:6F2zt7QZkdQKKBul5ATSMD',
      'track_href': 'https://api.spotify.com/v1/tracks/6F2zt7QZkdQKKBul5ATSMD',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6F2zt7QZkdQKKBul5ATSMD',
      'duration_ms': 150517,
      'time_signature': 4},
     {'danceability': 0.345,
      'energy': 0.786,
      'key': 7,
      'loudness': -5.632,
      'mode': 1,
      'speechiness': 0.0329,
      'acousticness': 0.207,
      'instrumentalness': 0,
      'liveness': 0.674,
      'valence': 0.86,
      'tempo': 172.259,
      'type': 'audio_features',
      'id': '4NwGrVIkJavdtdfX03hG0B',
      'uri': 'spotify:track:4NwGrVIkJavdtdfX03hG0B',
      'track_href': 'https://api.spotify.com/v1/tracks/4NwGrVIkJavdtdfX03hG0B',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4NwGrVIkJavdtdfX03hG0B',
      'duration_ms': 169867,
      'time_signature': 4},
     {'danceability': 0.623,
      'energy': 0.692,
      'key': 9,
      'loudness': -7.977,
      'mode': 1,
      'speechiness': 0.0293,
      'acousticness': 0.0173,
      'instrumentalness': 0.00883,
      'liveness': 0.124,
      'valence': 0.13,
      'tempo': 104.977,
      'type': 'audio_features',
      'id': '5KzuAU7zxcP0bq0CPdRRyr',
      'uri': 'spotify:track:5KzuAU7zxcP0bq0CPdRRyr',
      'track_href': 'https://api.spotify.com/v1/tracks/5KzuAU7zxcP0bq0CPdRRyr',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5KzuAU7zxcP0bq0CPdRRyr',
      'duration_ms': 202840,
      'time_signature': 4},
     {'danceability': 0.513,
      'energy': 0.84,
      'key': 2,
      'loudness': -6.07,
      'mode': 1,
      'speechiness': 0.0321,
      'acousticness': 0.183,
      'instrumentalness': 0.0546,
      'liveness': 0.106,
      'valence': 0.269,
      'tempo': 95.048,
      'type': 'audio_features',
      'id': '5XvS3t5O7c9X8cSoIIp3At',
      'uri': 'spotify:track:5XvS3t5O7c9X8cSoIIp3At',
      'track_href': 'https://api.spotify.com/v1/tracks/5XvS3t5O7c9X8cSoIIp3At',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5XvS3t5O7c9X8cSoIIp3At',
      'duration_ms': 260067,
      'time_signature': 4},
     {'danceability': 0.635,
      'energy': 0.358,
      'key': 9,
      'loudness': -10.715,
      'mode': 0,
      'speechiness': 0.0551,
      'acousticness': 0.575,
      'instrumentalness': 0.000196,
      'liveness': 0.0961,
      'valence': 0.254,
      'tempo': 134.904,
      'type': 'audio_features',
      'id': '7IaKUljpLoG5eDfEklNM9x',
      'uri': 'spotify:track:7IaKUljpLoG5eDfEklNM9x',
      'track_href': 'https://api.spotify.com/v1/tracks/7IaKUljpLoG5eDfEklNM9x',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7IaKUljpLoG5eDfEklNM9x',
      'duration_ms': 268547,
      'time_signature': 4},
     {'danceability': 0.372,
      'energy': 0.798,
      'key': 7,
      'loudness': -7.835,
      'mode': 1,
      'speechiness': 0.0554,
      'acousticness': 0.734,
      'instrumentalness': 0,
      'liveness': 0.924,
      'valence': 0.851,
      'tempo': 102.722,
      'type': 'audio_features',
      'id': '3g0zYe7PwBeF0iPiYFCSfu',
      'uri': 'spotify:track:3g0zYe7PwBeF0iPiYFCSfu',
      'track_href': 'https://api.spotify.com/v1/tracks/3g0zYe7PwBeF0iPiYFCSfu',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3g0zYe7PwBeF0iPiYFCSfu',
      'duration_ms': 131293,
      'time_signature': 4},
     {'danceability': 0.622,
      'energy': 0.758,
      'key': 0,
      'loudness': -5.384,
      'mode': 1,
      'speechiness': 0.0603,
      'acousticness': 0.0132,
      'instrumentalness': 0,
      'liveness': 0.0615,
      'valence': 0.327,
      'tempo': 119.927,
      'type': 'audio_features',
      'id': '4wPbR6XonWB7fiyWUMAaH2',
      'uri': 'spotify:track:4wPbR6XonWB7fiyWUMAaH2',
      'track_href': 'https://api.spotify.com/v1/tracks/4wPbR6XonWB7fiyWUMAaH2',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4wPbR6XonWB7fiyWUMAaH2',
      'duration_ms': 179307,
      'time_signature': 4},
     {'danceability': 0.515,
      'energy': 0.397,
      'key': 0,
      'loudness': -7.661,
      'mode': 1,
      'speechiness': 0.028,
      'acousticness': 0.642,
      'instrumentalness': 3.33e-06,
      'liveness': 0.146,
      'valence': 0.365,
      'tempo': 77.527,
      'type': 'audio_features',
      'id': '2aOmlow495KuwYCcT8ZD4l',
      'uri': 'spotify:track:2aOmlow495KuwYCcT8ZD4l',
      'track_href': 'https://api.spotify.com/v1/tracks/2aOmlow495KuwYCcT8ZD4l',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/2aOmlow495KuwYCcT8ZD4l',
      'duration_ms': 203121,
      'time_signature': 1},
     {'danceability': 0.385,
      'energy': 0.235,
      'key': 1,
      'loudness': -18.742,
      'mode': 1,
      'speechiness': 0.0348,
      'acousticness': 0.987,
      'instrumentalness': 0.898,
      'liveness': 0.193,
      'valence': 0.0615,
      'tempo': 110.048,
      'type': 'audio_features',
      'id': '6EmP2GA5oTASX7I2VWsHW0',
      'uri': 'spotify:track:6EmP2GA5oTASX7I2VWsHW0',
      'track_href': 'https://api.spotify.com/v1/tracks/6EmP2GA5oTASX7I2VWsHW0',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6EmP2GA5oTASX7I2VWsHW0',
      'duration_ms': 246907,
      'time_signature': 4},
     {'danceability': 0.727,
      'energy': 0.718,
      'key': 1,
      'loudness': -7.453,
      'mode': 1,
      'speechiness': 0.298,
      'acousticness': 0.0388,
      'instrumentalness': 0.0262,
      'liveness': 0.328,
      'valence': 0.15,
      'tempo': 125.975,
      'type': 'audio_features',
      'id': '7zOVwLxNKAG4FNNSKgJUv6',
      'uri': 'spotify:track:7zOVwLxNKAG4FNNSKgJUv6',
      'track_href': 'https://api.spotify.com/v1/tracks/7zOVwLxNKAG4FNNSKgJUv6',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7zOVwLxNKAG4FNNSKgJUv6',
      'duration_ms': 250476,
      'time_signature': 4},
     {'danceability': 0.572,
      'energy': 0.752,
      'key': 7,
      'loudness': -8.179,
      'mode': 1,
      'speechiness': 0.0313,
      'acousticness': 0.0216,
      'instrumentalness': 0.000133,
      'liveness': 0.103,
      'valence': 0.521,
      'tempo': 118.309,
      'type': 'audio_features',
      'id': '4A0IWubd7pVbtfQVL5Zw7V',
      'uri': 'spotify:track:4A0IWubd7pVbtfQVL5Zw7V',
      'track_href': 'https://api.spotify.com/v1/tracks/4A0IWubd7pVbtfQVL5Zw7V',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4A0IWubd7pVbtfQVL5Zw7V',
      'duration_ms': 143280,
      'time_signature': 4},
     {'danceability': 0.758,
      'energy': 0.896,
      'key': 9,
      'loudness': -3.311,
      'mode': 1,
      'speechiness': 0.0501,
      'acousticness': 0.173,
      'instrumentalness': 3.18e-05,
      'liveness': 0.136,
      'valence': 0.797,
      'tempo': 94.911,
      'type': 'audio_features',
      'id': '460Wn6Dq2uMviG5nPXtPnb',
      'uri': 'spotify:track:460Wn6Dq2uMviG5nPXtPnb',
      'track_href': 'https://api.spotify.com/v1/tracks/460Wn6Dq2uMviG5nPXtPnb',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/460Wn6Dq2uMviG5nPXtPnb',
      'duration_ms': 207027,
      'time_signature': 4},
     {'danceability': 0.401,
      'energy': 0.134,
      'key': 3,
      'loudness': -11.205,
      'mode': 1,
      'speechiness': 0.0317,
      'acousticness': 0.937,
      'instrumentalness': 0,
      'liveness': 0.109,
      'valence': 0.296,
      'tempo': 62.72,
      'type': 'audio_features',
      'id': '6phxaP5jCXWiC1dBWn9vIT',
      'uri': 'spotify:track:6phxaP5jCXWiC1dBWn9vIT',
      'track_href': 'https://api.spotify.com/v1/tracks/6phxaP5jCXWiC1dBWn9vIT',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6phxaP5jCXWiC1dBWn9vIT',
      'duration_ms': 218120,
      'time_signature': 4},
     {'danceability': 0.896,
      'energy': 0.552,
      'key': 4,
      'loudness': -6.11,
      'mode': 0,
      'speechiness': 0.102,
      'acousticness': 0.273,
      'instrumentalness': 0,
      'liveness': 0.0662,
      'valence': 0.657,
      'tempo': 93.001,
      'type': 'audio_features',
      'id': '1T1ZUKX4X87tVLaBGjwFv4',
      'uri': 'spotify:track:1T1ZUKX4X87tVLaBGjwFv4',
      'track_href': 'https://api.spotify.com/v1/tracks/1T1ZUKX4X87tVLaBGjwFv4',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1T1ZUKX4X87tVLaBGjwFv4',
      'duration_ms': 237840,
      'time_signature': 4},
     {'danceability': 0.762,
      'energy': 0.835,
      'key': 1,
      'loudness': -6.789,
      'mode': 0,
      'speechiness': 0.126,
      'acousticness': 0.0372,
      'instrumentalness': 0,
      'liveness': 0.102,
      'valence': 0.822,
      'tempo': 127.001,
      'type': 'audio_features',
      'id': '6wJS2e5H75mgshQtEMMdyR',
      'uri': 'spotify:track:6wJS2e5H75mgshQtEMMdyR',
      'track_href': 'https://api.spotify.com/v1/tracks/6wJS2e5H75mgshQtEMMdyR',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6wJS2e5H75mgshQtEMMdyR',
      'duration_ms': 208253,
      'time_signature': 4},
     {'danceability': 0.62,
      'energy': 0.755,
      'key': 10,
      'loudness': -4.767,
      'mode': 0,
      'speechiness': 0.296,
      'acousticness': 0.172,
      'instrumentalness': 0,
      'liveness': 0.26,
      'valence': 0.22,
      'tempo': 89.433,
      'type': 'audio_features',
      'id': '7lat1bl4iCu4y2J6GsG4k6',
      'uri': 'spotify:track:7lat1bl4iCu4y2J6GsG4k6',
      'track_href': 'https://api.spotify.com/v1/tracks/7lat1bl4iCu4y2J6GsG4k6',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7lat1bl4iCu4y2J6GsG4k6',
      'duration_ms': 146107,
      'time_signature': 4},
     {'danceability': 0.448,
      'energy': 0.145,
      'key': 0,
      'loudness': -12.256,
      'mode': 1,
      'speechiness': 0.0305,
      'acousticness': 0.734,
      'instrumentalness': 5.48e-05,
      'liveness': 0.0845,
      'valence': 0.185,
      'tempo': 115.29,
      'type': 'audio_features',
      'id': '6hpPuPBDSYoqEbuO7QVNZ6',
      'uri': 'spotify:track:6hpPuPBDSYoqEbuO7QVNZ6',
      'track_href': 'https://api.spotify.com/v1/tracks/6hpPuPBDSYoqEbuO7QVNZ6',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6hpPuPBDSYoqEbuO7QVNZ6',
      'duration_ms': 328800,
      'time_signature': 3},
     {'danceability': 0.74,
      'energy': 0.871,
      'key': 10,
      'loudness': -7.871,
      'mode': 0,
      'speechiness': 0.0507,
      'acousticness': 0.208,
      'instrumentalness': 0.000127,
      'liveness': 0.0917,
      'valence': 0.541,
      'tempo': 106.694,
      'type': 'audio_features',
      'id': '6fsLjItlUmbpl16SGi2COD',
      'uri': 'spotify:track:6fsLjItlUmbpl16SGi2COD',
      'track_href': 'https://api.spotify.com/v1/tracks/6fsLjItlUmbpl16SGi2COD',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6fsLjItlUmbpl16SGi2COD',
      'duration_ms': 828360,
      'time_signature': 4},
     {'danceability': 0.507,
      'energy': 0.718,
      'key': 2,
      'loudness': -6.415,
      'mode': 1,
      'speechiness': 0.025,
      'acousticness': 0.0194,
      'instrumentalness': 0.00914,
      'liveness': 0.135,
      'valence': 0.824,
      'tempo': 161.12,
      'type': 'audio_features',
      'id': '3DIEoK7wpt2j7ECiTEZxKr',
      'uri': 'spotify:track:3DIEoK7wpt2j7ECiTEZxKr',
      'track_href': 'https://api.spotify.com/v1/tracks/3DIEoK7wpt2j7ECiTEZxKr',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3DIEoK7wpt2j7ECiTEZxKr',
      'duration_ms': 205133,
      'time_signature': 4},
     {'danceability': 0.297,
      'energy': 0.284,
      'key': 6,
      'loudness': -13.283,
      'mode': 1,
      'speechiness': 0.0442,
      'acousticness': 0.42,
      'instrumentalness': 0.0838,
      'liveness': 0.0959,
      'valence': 0.172,
      'tempo': 165.668,
      'type': 'audio_features',
      'id': '3ieOG3Yngp40fxrLO7wEVu',
      'uri': 'spotify:track:3ieOG3Yngp40fxrLO7wEVu',
      'track_href': 'https://api.spotify.com/v1/tracks/3ieOG3Yngp40fxrLO7wEVu',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3ieOG3Yngp40fxrLO7wEVu',
      'duration_ms': 192580,
      'time_signature': 4},
     {'danceability': 0.852,
      'energy': 0.881,
      'key': 2,
      'loudness': -9.086,
      'mode': 1,
      'speechiness': 0.0618,
      'acousticness': 0.464,
      'instrumentalness': 0,
      'liveness': 0.264,
      'valence': 0.796,
      'tempo': 106.598,
      'type': 'audio_features',
      'id': '5qZagGLQhXUL7osnIwkWAc',
      'uri': 'spotify:track:5qZagGLQhXUL7osnIwkWAc',
      'track_href': 'https://api.spotify.com/v1/tracks/5qZagGLQhXUL7osnIwkWAc',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5qZagGLQhXUL7osnIwkWAc',
      'duration_ms': 184373,
      'time_signature': 4},
     {'danceability': 0.673,
      'energy': 0.479,
      'key': 6,
      'loudness': -5.825,
      'mode': 1,
      'speechiness': 0.025,
      'acousticness': 0.818,
      'instrumentalness': 0,
      'liveness': 0.254,
      'valence': 0.295,
      'tempo': 105.049,
      'type': 'audio_features',
      'id': '268qdjAVksx6iRKNFrQ9v6',
      'uri': 'spotify:track:268qdjAVksx6iRKNFrQ9v6',
      'track_href': 'https://api.spotify.com/v1/tracks/268qdjAVksx6iRKNFrQ9v6',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/268qdjAVksx6iRKNFrQ9v6',
      'duration_ms': 315067,
      'time_signature': 4},
     {'danceability': 0.687,
      'energy': 0.645,
      'key': 7,
      'loudness': -8.85,
      'mode': 1,
      'speechiness': 0.0267,
      'acousticness': 0.00114,
      'instrumentalness': 0.000104,
      'liveness': 0.273,
      'valence': 0.726,
      'tempo': 125.012,
      'type': 'audio_features',
      'id': '6pKauRPBN29rgOH533Hilw',
      'uri': 'spotify:track:6pKauRPBN29rgOH533Hilw',
      'track_href': 'https://api.spotify.com/v1/tracks/6pKauRPBN29rgOH533Hilw',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6pKauRPBN29rgOH533Hilw',
      'duration_ms': 258120,
      'time_signature': 4},
     {'danceability': 0.711,
      'energy': 0.758,
      'key': 9,
      'loudness': -8.385,
      'mode': 1,
      'speechiness': 0.0326,
      'acousticness': 0.162,
      'instrumentalness': 0.011,
      'liveness': 0.185,
      'valence': 0.842,
      'tempo': 115.367,
      'type': 'audio_features',
      'id': '0jasbGRELTM14tJi9SxGGF',
      'uri': 'spotify:track:0jasbGRELTM14tJi9SxGGF',
      'track_href': 'https://api.spotify.com/v1/tracks/0jasbGRELTM14tJi9SxGGF',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0jasbGRELTM14tJi9SxGGF',
      'duration_ms': 185467,
      'time_signature': 4},
     {'danceability': 0.625,
      'energy': 0.741,
      'key': 11,
      'loudness': -7.817,
      'mode': 1,
      'speechiness': 0.042,
      'acousticness': 0.0532,
      'instrumentalness': 0.00255,
      'liveness': 0.0916,
      'valence': 0.177,
      'tempo': 121.017,
      'type': 'audio_features',
      'id': '5lY4RqjXhXwinOSoNunJhj',
      'uri': 'spotify:track:5lY4RqjXhXwinOSoNunJhj',
      'track_href': 'https://api.spotify.com/v1/tracks/5lY4RqjXhXwinOSoNunJhj',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5lY4RqjXhXwinOSoNunJhj',
      'duration_ms': 231424,
      'time_signature': 4},
     {'danceability': 0.752,
      'energy': 0.207,
      'key': 0,
      'loudness': -10.284,
      'mode': 0,
      'speechiness': 0.0432,
      'acousticness': 0.442,
      'instrumentalness': 0,
      'liveness': 0.159,
      'valence': 0.647,
      'tempo': 134.88,
      'type': 'audio_features',
      'id': '0dTTL0bKQKtEYdsia2Vnuv',
      'uri': 'spotify:track:0dTTL0bKQKtEYdsia2Vnuv',
      'track_href': 'https://api.spotify.com/v1/tracks/0dTTL0bKQKtEYdsia2Vnuv',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0dTTL0bKQKtEYdsia2Vnuv',
      'duration_ms': 157333,
      'time_signature': 4},
     {'danceability': 0.438,
      'energy': 0.822,
      'key': 2,
      'loudness': -6.618,
      'mode': 0,
      'speechiness': 0.0503,
      'acousticness': 0.000102,
      'instrumentalness': 0.000124,
      'liveness': 0.337,
      'valence': 0.26,
      'tempo': 123.969,
      'type': 'audio_features',
      'id': '7c67VdAcd7DzEfzzBWvjDE',
      'uri': 'spotify:track:7c67VdAcd7DzEfzzBWvjDE',
      'track_href': 'https://api.spotify.com/v1/tracks/7c67VdAcd7DzEfzzBWvjDE',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7c67VdAcd7DzEfzzBWvjDE',
      'duration_ms': 268093,
      'time_signature': 4},
     {'danceability': 0.682,
      'energy': 0.597,
      'key': 3,
      'loudness': -8.866,
      'mode': 1,
      'speechiness': 0.0405,
      'acousticness': 0.564,
      'instrumentalness': 0,
      'liveness': 0.331,
      'valence': 0.465,
      'tempo': 138.09,
      'type': 'audio_features',
      'id': '4c2Zdtqy5x9BkhTSJRTDmk',
      'uri': 'spotify:track:4c2Zdtqy5x9BkhTSJRTDmk',
      'track_href': 'https://api.spotify.com/v1/tracks/4c2Zdtqy5x9BkhTSJRTDmk',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4c2Zdtqy5x9BkhTSJRTDmk',
      'duration_ms': 223680,
      'time_signature': 4},
     {'danceability': 0.677,
      'energy': 0.76,
      'key': 10,
      'loudness': -7.659,
      'mode': 0,
      'speechiness': 0.0962,
      'acousticness': 0.00257,
      'instrumentalness': 0.000766,
      'liveness': 0.303,
      'valence': 0.57,
      'tempo': 99.795,
      'type': 'audio_features',
      'id': '0cfn5OBGafl32fEsc3z4GE',
      'uri': 'spotify:track:0cfn5OBGafl32fEsc3z4GE',
      'track_href': 'https://api.spotify.com/v1/tracks/0cfn5OBGafl32fEsc3z4GE',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0cfn5OBGafl32fEsc3z4GE',
      'duration_ms': 272640,
      'time_signature': 4},
     {'danceability': 0.503,
      'energy': 0.791,
      'key': 0,
      'loudness': -5.31,
      'mode': 1,
      'speechiness': 0.0346,
      'acousticness': 0.0596,
      'instrumentalness': 0,
      'liveness': 0.0927,
      'valence': 0.519,
      'tempo': 141.825,
      'type': 'audio_features',
      'id': '5H78WooTqR4bbH85MNKnSD',
      'uri': 'spotify:track:5H78WooTqR4bbH85MNKnSD',
      'track_href': 'https://api.spotify.com/v1/tracks/5H78WooTqR4bbH85MNKnSD',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5H78WooTqR4bbH85MNKnSD',
      'duration_ms': 199227,
      'time_signature': 4},
     {'danceability': 0.488,
      'energy': 0.95,
      'key': 8,
      'loudness': -10.774,
      'mode': 1,
      'speechiness': 0.0668,
      'acousticness': 0.00493,
      'instrumentalness': 0,
      'liveness': 0.344,
      'valence': 0.428,
      'tempo': 124.823,
      'type': 'audio_features',
      'id': '3laUFNdKR36ecG4WheN1bm',
      'uri': 'spotify:track:3laUFNdKR36ecG4WheN1bm',
      'track_href': 'https://api.spotify.com/v1/tracks/3laUFNdKR36ecG4WheN1bm',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3laUFNdKR36ecG4WheN1bm',
      'duration_ms': 102373,
      'time_signature': 4},
     {'danceability': 0.659,
      'energy': 0.873,
      'key': 0,
      'loudness': -5.168,
      'mode': 0,
      'speechiness': 0.0846,
      'acousticness': 0.139,
      'instrumentalness': 2.75e-06,
      'liveness': 0.528,
      'valence': 0.209,
      'tempo': 140.124,
      'type': 'audio_features',
      'id': '7wVDl1ITx75ifEyNacj8Gm',
      'uri': 'spotify:track:7wVDl1ITx75ifEyNacj8Gm',
      'track_href': 'https://api.spotify.com/v1/tracks/7wVDl1ITx75ifEyNacj8Gm',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7wVDl1ITx75ifEyNacj8Gm',
      'duration_ms': 205507,
      'time_signature': 4},
     {'danceability': 0.806,
      'energy': 0.756,
      'key': 11,
      'loudness': -3.63,
      'mode': 1,
      'speechiness': 0.0418,
      'acousticness': 0.514,
      'instrumentalness': 0,
      'liveness': 0.129,
      'valence': 0.819,
      'tempo': 122.014,
      'type': 'audio_features',
      'id': '0nwKnR1knaufGh13KAvSPO',
      'uri': 'spotify:track:0nwKnR1knaufGh13KAvSPO',
      'track_href': 'https://api.spotify.com/v1/tracks/0nwKnR1knaufGh13KAvSPO',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0nwKnR1knaufGh13KAvSPO',
      'duration_ms': 276867,
      'time_signature': 4},
     {'danceability': 0.604,
      'energy': 0.538,
      'key': 4,
      'loudness': -11.04,
      'mode': 1,
      'speechiness': 0.0843,
      'acousticness': 0.85,
      'instrumentalness': 0.00139,
      'liveness': 0.101,
      'valence': 0.157,
      'tempo': 151.955,
      'type': 'audio_features',
      'id': '5EaV2esdrpepimHDkZJ303',
      'uri': 'spotify:track:5EaV2esdrpepimHDkZJ303',
      'track_href': 'https://api.spotify.com/v1/tracks/5EaV2esdrpepimHDkZJ303',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5EaV2esdrpepimHDkZJ303',
      'duration_ms': 189103,
      'time_signature': 4},
     {'danceability': 0.0913,
      'energy': 0.93,
      'key': 7,
      'loudness': -6.34,
      'mode': 1,
      'speechiness': 0.115,
      'acousticness': 3.26e-05,
      'instrumentalness': 0.79,
      'liveness': 0.0991,
      'valence': 0.0607,
      'tempo': 77.566,
      'type': 'audio_features',
      'id': '6aRjVN7fsSXisyETj3kGk8',
      'uri': 'spotify:track:6aRjVN7fsSXisyETj3kGk8',
      'track_href': 'https://api.spotify.com/v1/tracks/6aRjVN7fsSXisyETj3kGk8',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6aRjVN7fsSXisyETj3kGk8',
      'duration_ms': 333107,
      'time_signature': 4},
     {'danceability': 0.684,
      'energy': 0.693,
      'key': 1,
      'loudness': -8.093,
      'mode': 1,
      'speechiness': 0.0274,
      'acousticness': 0.0149,
      'instrumentalness': 0.171,
      'liveness': 0.127,
      'valence': 0.545,
      'tempo': 139.941,
      'type': 'audio_features',
      'id': '3iRfwDUIPauIdKMVUgDkSw',
      'uri': 'spotify:track:3iRfwDUIPauIdKMVUgDkSw',
      'track_href': 'https://api.spotify.com/v1/tracks/3iRfwDUIPauIdKMVUgDkSw',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3iRfwDUIPauIdKMVUgDkSw',
      'duration_ms': 129587,
      'time_signature': 4},
     {'danceability': 0.321,
      'energy': 0.737,
      'key': 2,
      'loudness': -4.264,
      'mode': 1,
      'speechiness': 0.0396,
      'acousticness': 0.0015,
      'instrumentalness': 6.87e-06,
      'liveness': 0.302,
      'valence': 0.0528,
      'tempo': 83.588,
      'type': 'audio_features',
      'id': '239VFYzJLnwfZbh6Fz29wY',
      'uri': 'spotify:track:239VFYzJLnwfZbh6Fz29wY',
      'track_href': 'https://api.spotify.com/v1/tracks/239VFYzJLnwfZbh6Fz29wY',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/239VFYzJLnwfZbh6Fz29wY',
      'duration_ms': 294320,
      'time_signature': 4},
     {'danceability': 0.707,
      'energy': 0.523,
      'key': 9,
      'loudness': -12.565,
      'mode': 1,
      'speechiness': 0.215,
      'acousticness': 0.687,
      'instrumentalness': 0.00168,
      'liveness': 0.385,
      'valence': 0.763,
      'tempo': 113.766,
      'type': 'audio_features',
      'id': '04ahAibOynzS5BD2m3KINz',
      'uri': 'spotify:track:04ahAibOynzS5BD2m3KINz',
      'track_href': 'https://api.spotify.com/v1/tracks/04ahAibOynzS5BD2m3KINz',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/04ahAibOynzS5BD2m3KINz',
      'duration_ms': 176200,
      'time_signature': 4},
     {'danceability': 0.824,
      'energy': 0.496,
      'key': 7,
      'loudness': -6.714,
      'mode': 1,
      'speechiness': 0.0448,
      'acousticness': 0.0346,
      'instrumentalness': 0.00538,
      'liveness': 0.121,
      'valence': 0.439,
      'tempo': 112.981,
      'type': 'audio_features',
      'id': '4z41UlGXD92A526CZgJT80',
      'uri': 'spotify:track:4z41UlGXD92A526CZgJT80',
      'track_href': 'https://api.spotify.com/v1/tracks/4z41UlGXD92A526CZgJT80',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4z41UlGXD92A526CZgJT80',
      'duration_ms': 147787,
      'time_signature': 4},
     {'danceability': 0.668,
      'energy': 0.746,
      'key': 11,
      'loudness': -6.647,
      'mode': 0,
      'speechiness': 0.329,
      'acousticness': 0.0343,
      'instrumentalness': 0,
      'liveness': 0.0939,
      'valence': 0.343,
      'tempo': 189.825,
      'type': 'audio_features',
      'id': '3jxpYEBewRHTW4koJoKBLB',
      'uri': 'spotify:track:3jxpYEBewRHTW4koJoKBLB',
      'track_href': 'https://api.spotify.com/v1/tracks/3jxpYEBewRHTW4koJoKBLB',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3jxpYEBewRHTW4koJoKBLB',
      'duration_ms': 166165,
      'time_signature': 4},
     {'danceability': 0.496,
      'energy': 0.761,
      'key': 1,
      'loudness': -5.651,
      'mode': 0,
      'speechiness': 0.0337,
      'acousticness': 0.000297,
      'instrumentalness': 1.66e-06,
      'liveness': 0.302,
      'valence': 0.355,
      'tempo': 94.972,
      'type': 'audio_features',
      'id': '1gwDIyATjLVi42oBsyUgyJ',
      'uri': 'spotify:track:1gwDIyATjLVi42oBsyUgyJ',
      'track_href': 'https://api.spotify.com/v1/tracks/1gwDIyATjLVi42oBsyUgyJ',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1gwDIyATjLVi42oBsyUgyJ',
      'duration_ms': 215560,
      'time_signature': 4},
     {'danceability': 0.369,
      'energy': 0.467,
      'key': 5,
      'loudness': -9.018,
      'mode': 1,
      'speechiness': 0.0274,
      'acousticness': 0.0194,
      'instrumentalness': 0.46,
      'liveness': 0.109,
      'valence': 0.174,
      'tempo': 94.473,
      'type': 'audio_features',
      'id': '3AVrVz5rK8Hrqo9YGiVGN5',
      'uri': 'spotify:track:3AVrVz5rK8Hrqo9YGiVGN5',
      'track_href': 'https://api.spotify.com/v1/tracks/3AVrVz5rK8Hrqo9YGiVGN5',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3AVrVz5rK8Hrqo9YGiVGN5',
      'duration_ms': 290147,
      'time_signature': 4},
     {'danceability': 0.647,
      'energy': 0.548,
      'key': 4,
      'loudness': -10.779,
      'mode': 1,
      'speechiness': 0.0664,
      'acousticness': 0.683,
      'instrumentalness': 0,
      'liveness': 0.0683,
      'valence': 0.536,
      'tempo': 102.624,
      'type': 'audio_features',
      'id': '2TBBZNf7Zq8QXWVp8E9Ip5',
      'uri': 'spotify:track:2TBBZNf7Zq8QXWVp8E9Ip5',
      'track_href': 'https://api.spotify.com/v1/tracks/2TBBZNf7Zq8QXWVp8E9Ip5',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/2TBBZNf7Zq8QXWVp8E9Ip5',
      'duration_ms': 218973,
      'time_signature': 4},
     {'danceability': 0.595,
      'energy': 0.759,
      'key': 7,
      'loudness': -11.162,
      'mode': 1,
      'speechiness': 0.102,
      'acousticness': 0.484,
      'instrumentalness': 1.18e-05,
      'liveness': 0.384,
      'valence': 0.637,
      'tempo': 114.207,
      'type': 'audio_features',
      'id': '1f9FEeBWZkuT4ItJZwsJ2l',
      'uri': 'spotify:track:1f9FEeBWZkuT4ItJZwsJ2l',
      'track_href': 'https://api.spotify.com/v1/tracks/1f9FEeBWZkuT4ItJZwsJ2l',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1f9FEeBWZkuT4ItJZwsJ2l',
      'duration_ms': 156893,
      'time_signature': 4},
     {'danceability': 0.632,
      'energy': 0.731,
      'key': 1,
      'loudness': -4.191,
      'mode': 1,
      'speechiness': 0.262,
      'acousticness': 0.0826,
      'instrumentalness': 0,
      'liveness': 0.0514,
      'valence': 0.785,
      'tempo': 89.483,
      'type': 'audio_features',
      'id': '3j1UrSXCeWYa5ltei4ZAxt',
      'uri': 'spotify:track:3j1UrSXCeWYa5ltei4ZAxt',
      'track_href': 'https://api.spotify.com/v1/tracks/3j1UrSXCeWYa5ltei4ZAxt',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3j1UrSXCeWYa5ltei4ZAxt',
      'duration_ms': 238680,
      'time_signature': 4},
     {'danceability': 0.787,
      'energy': 0.452,
      'key': 11,
      'loudness': -7.381,
      'mode': 0,
      'speechiness': 0.0314,
      'acousticness': 0.622,
      'instrumentalness': 0,
      'liveness': 0.134,
      'valence': 0.773,
      'tempo': 118.982,
      'type': 'audio_features',
      'id': '2OOK1heX7PHmAzd2qfzQP6',
      'uri': 'spotify:track:2OOK1heX7PHmAzd2qfzQP6',
      'track_href': 'https://api.spotify.com/v1/tracks/2OOK1heX7PHmAzd2qfzQP6',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/2OOK1heX7PHmAzd2qfzQP6',
      'duration_ms': 241453,
      'time_signature': 4},
     {'danceability': 0.666,
      'energy': 0.738,
      'key': 1,
      'loudness': -6.86,
      'mode': 0,
      'speechiness': 0.04,
      'acousticness': 0.623,
      'instrumentalness': 0.00156,
      'liveness': 0.0744,
      'valence': 0.884,
      'tempo': 95.994,
      'type': 'audio_features',
      'id': '0v2WItKxInx9szDEBVwDri',
      'uri': 'spotify:track:0v2WItKxInx9szDEBVwDri',
      'track_href': 'https://api.spotify.com/v1/tracks/0v2WItKxInx9szDEBVwDri',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0v2WItKxInx9szDEBVwDri',
      'duration_ms': 189940,
      'time_signature': 4},
     {'danceability': 0.47,
      'energy': 0.825,
      'key': 4,
      'loudness': -4.562,
      'mode': 1,
      'speechiness': 0.235,
      'acousticness': 0.0616,
      'instrumentalness': 0.000216,
      'liveness': 0.104,
      'valence': 0.323,
      'tempo': 166.286,
      'type': 'audio_features',
      'id': '1MY4HVXfBkUg1mUjmCjznI',
      'uri': 'spotify:track:1MY4HVXfBkUg1mUjmCjznI',
      'track_href': 'https://api.spotify.com/v1/tracks/1MY4HVXfBkUg1mUjmCjznI',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1MY4HVXfBkUg1mUjmCjznI',
      'duration_ms': 379947,
      'time_signature': 4},
     {'danceability': 0.626,
      'energy': 0.739,
      'key': 8,
      'loudness': -5.198,
      'mode': 1,
      'speechiness': 0.0357,
      'acousticness': 0.000841,
      'instrumentalness': 2.01e-05,
      'liveness': 0.123,
      'valence': 0.424,
      'tempo': 106.999,
      'type': 'audio_features',
      'id': '0aoC5tywVB3Rj11z4ihvK4',
      'uri': 'spotify:track:0aoC5tywVB3Rj11z4ihvK4',
      'track_href': 'https://api.spotify.com/v1/tracks/0aoC5tywVB3Rj11z4ihvK4',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0aoC5tywVB3Rj11z4ihvK4',
      'duration_ms': 222853,
      'time_signature': 4},
     {'danceability': 0.412,
      'energy': 0.964,
      'key': 6,
      'loudness': -7.557,
      'mode': 0,
      'speechiness': 0.169,
      'acousticness': 1.84e-06,
      'instrumentalness': 0.146,
      'liveness': 0.502,
      'valence': 0.232,
      'tempo': 142.536,
      'type': 'audio_features',
      'id': '4MVJvkKhoeLPH3BcssmVpC',
      'uri': 'spotify:track:4MVJvkKhoeLPH3BcssmVpC',
      'track_href': 'https://api.spotify.com/v1/tracks/4MVJvkKhoeLPH3BcssmVpC',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4MVJvkKhoeLPH3BcssmVpC',
      'duration_ms': 293347,
      'time_signature': 3},
     {'danceability': 0.894,
      'energy': 0.879,
      'key': 6,
      'loudness': -5.023,
      'mode': 0,
      'speechiness': 0.26,
      'acousticness': 0.168,
      'instrumentalness': 0,
      'liveness': 0.0851,
      'valence': 0.885,
      'tempo': 101.759,
      'type': 'audio_features',
      'id': '6XKRP0dq5PBVJROcKa9NnJ',
      'uri': 'spotify:track:6XKRP0dq5PBVJROcKa9NnJ',
      'track_href': 'https://api.spotify.com/v1/tracks/6XKRP0dq5PBVJROcKa9NnJ',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6XKRP0dq5PBVJROcKa9NnJ',
      'duration_ms': 194973,
      'time_signature': 4},
     {'danceability': 0.667,
      'energy': 0.808,
      'key': 1,
      'loudness': -4.667,
      'mode': 1,
      'speechiness': 0.265,
      'acousticness': 0.141,
      'instrumentalness': 0,
      'liveness': 0.115,
      'valence': 0.653,
      'tempo': 83.031,
      'type': 'audio_features',
      'id': '76QWmZ9lR65gGfe03jmmPP',
      'uri': 'spotify:track:76QWmZ9lR65gGfe03jmmPP',
      'track_href': 'https://api.spotify.com/v1/tracks/76QWmZ9lR65gGfe03jmmPP',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/76QWmZ9lR65gGfe03jmmPP',
      'duration_ms': 265013,
      'time_signature': 4},
     {'danceability': 0.708,
      'energy': 0.538,
      'key': 1,
      'loudness': -13.356,
      'mode': 0,
      'speechiness': 0.0975,
      'acousticness': 0.00332,
      'instrumentalness': 0,
      'liveness': 0.0728,
      'valence': 0.844,
      'tempo': 135.953,
      'type': 'audio_features',
      'id': '4pzeuvQ6REsnqhm5OjEhNy',
      'uri': 'spotify:track:4pzeuvQ6REsnqhm5OjEhNy',
      'track_href': 'https://api.spotify.com/v1/tracks/4pzeuvQ6REsnqhm5OjEhNy',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4pzeuvQ6REsnqhm5OjEhNy',
      'duration_ms': 293268,
      'time_signature': 4},
     {'danceability': 0.52,
      'energy': 0.843,
      'key': 10,
      'loudness': -4.711,
      'mode': 0,
      'speechiness': 0.09,
      'acousticness': 0.00662,
      'instrumentalness': 1.39e-06,
      'liveness': 0.152,
      'valence': 0.521,
      'tempo': 129.984,
      'type': 'audio_features',
      'id': '1tUiPKYOob0YbMdVRbr79w',
      'uri': 'spotify:track:1tUiPKYOob0YbMdVRbr79w',
      'track_href': 'https://api.spotify.com/v1/tracks/1tUiPKYOob0YbMdVRbr79w',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1tUiPKYOob0YbMdVRbr79w',
      'duration_ms': 247347,
      'time_signature': 4},
     {'danceability': 0.58,
      'energy': 0.666,
      'key': 11,
      'loudness': -13.372,
      'mode': 1,
      'speechiness': 0.035,
      'acousticness': 0.223,
      'instrumentalness': 0.901,
      'liveness': 0.114,
      'valence': 0.11,
      'tempo': 136.001,
      'type': 'audio_features',
      'id': '6tSQ3Ds8TkjchyvLNgveVy',
      'uri': 'spotify:track:6tSQ3Ds8TkjchyvLNgveVy',
      'track_href': 'https://api.spotify.com/v1/tracks/6tSQ3Ds8TkjchyvLNgveVy',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6tSQ3Ds8TkjchyvLNgveVy',
      'duration_ms': 175120,
      'time_signature': 5},
     {'danceability': 0.747,
      'energy': 0.782,
      'key': 8,
      'loudness': -4.326,
      'mode': 1,
      'speechiness': 0.283,
      'acousticness': 0.0223,
      'instrumentalness': 0,
      'liveness': 0.0902,
      'valence': 0.501,
      'tempo': 113.99,
      'type': 'audio_features',
      'id': '31RYcUDhkqkH1W2xxnzBjY',
      'uri': 'spotify:track:31RYcUDhkqkH1W2xxnzBjY',
      'track_href': 'https://api.spotify.com/v1/tracks/31RYcUDhkqkH1W2xxnzBjY',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/31RYcUDhkqkH1W2xxnzBjY',
      'duration_ms': 206316,
      'time_signature': 4},
     {'danceability': 0.634,
      'energy': 0.667,
      'key': 6,
      'loudness': -8.189,
      'mode': 1,
      'speechiness': 0.0617,
      'acousticness': 0.338,
      'instrumentalness': 4.29e-06,
      'liveness': 0.0788,
      'valence': 0.455,
      'tempo': 96.01,
      'type': 'audio_features',
      'id': '6urBMNarztXhgZ93wItFTQ',
      'uri': 'spotify:track:6urBMNarztXhgZ93wItFTQ',
      'track_href': 'https://api.spotify.com/v1/tracks/6urBMNarztXhgZ93wItFTQ',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6urBMNarztXhgZ93wItFTQ',
      'duration_ms': 192610,
      'time_signature': 4},
     {'danceability': 0.507,
      'energy': 0.42,
      'key': 0,
      'loudness': -7.177,
      'mode': 1,
      'speechiness': 0.0282,
      'acousticness': 0.242,
      'instrumentalness': 0,
      'liveness': 0.122,
      'valence': 0.282,
      'tempo': 137.952,
      'type': 'audio_features',
      'id': '3IjTEp4wLCr0WVyJkvg1kj',
      'uri': 'spotify:track:3IjTEp4wLCr0WVyJkvg1kj',
      'track_href': 'https://api.spotify.com/v1/tracks/3IjTEp4wLCr0WVyJkvg1kj',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3IjTEp4wLCr0WVyJkvg1kj',
      'duration_ms': 271361,
      'time_signature': 4},
     {'danceability': 0.766,
      'energy': 0.741,
      'key': 2,
      'loudness': -5.642,
      'mode': 1,
      'speechiness': 0.213,
      'acousticness': 0.0858,
      'instrumentalness': 0,
      'liveness': 0.0991,
      'valence': 0.57,
      'tempo': 110.05,
      'type': 'audio_features',
      'id': '6SE1opPh1fvqgOSt0leF0d',
      'uri': 'spotify:track:6SE1opPh1fvqgOSt0leF0d',
      'track_href': 'https://api.spotify.com/v1/tracks/6SE1opPh1fvqgOSt0leF0d',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6SE1opPh1fvqgOSt0leF0d',
      'duration_ms': 204957,
      'time_signature': 4},
     {'danceability': 0.486,
      'energy': 0.633,
      'key': 2,
      'loudness': -8.64,
      'mode': 1,
      'speechiness': 0.038,
      'acousticness': 0.0478,
      'instrumentalness': 0,
      'liveness': 0.261,
      'valence': 0.238,
      'tempo': 127.979,
      'type': 'audio_features',
      'id': '7Ht9ePi78nxKAinLC0QVe2',
      'uri': 'spotify:track:7Ht9ePi78nxKAinLC0QVe2',
      'track_href': 'https://api.spotify.com/v1/tracks/7Ht9ePi78nxKAinLC0QVe2',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7Ht9ePi78nxKAinLC0QVe2',
      'duration_ms': 193187,
      'time_signature': 4},
     {'danceability': 0.455,
      'energy': 0.944,
      'key': 1,
      'loudness': -2.562,
      'mode': 1,
      'speechiness': 0.0619,
      'acousticness': 2.56e-06,
      'instrumentalness': 0.885,
      'liveness': 0.152,
      'valence': 0.137,
      'tempo': 99.985,
      'type': 'audio_features',
      'id': '06JJ9AUDnBRJPTYRCQk3mF',
      'uri': 'spotify:track:06JJ9AUDnBRJPTYRCQk3mF',
      'track_href': 'https://api.spotify.com/v1/tracks/06JJ9AUDnBRJPTYRCQk3mF',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/06JJ9AUDnBRJPTYRCQk3mF',
      'duration_ms': 136093,
      'time_signature': 4},
     {'danceability': 0.54,
      'energy': 0.643,
      'key': 11,
      'loudness': -9.088,
      'mode': 0,
      'speechiness': 0.0485,
      'acousticness': 0.302,
      'instrumentalness': 0.000135,
      'liveness': 0.0881,
      'valence': 0.593,
      'tempo': 159.929,
      'type': 'audio_features',
      'id': '6OXt9aSIr4DSxSR3Qjrtgp',
      'uri': 'spotify:track:6OXt9aSIr4DSxSR3Qjrtgp',
      'track_href': 'https://api.spotify.com/v1/tracks/6OXt9aSIr4DSxSR3Qjrtgp',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6OXt9aSIr4DSxSR3Qjrtgp',
      'duration_ms': 314853,
      'time_signature': 4},
     {'danceability': 0.334,
      'energy': 0.99,
      'key': 1,
      'loudness': -3.961,
      'mode': 1,
      'speechiness': 0.187,
      'acousticness': 0.000779,
      'instrumentalness': 0,
      'liveness': 0.0744,
      'valence': 0.395,
      'tempo': 145.162,
      'type': 'audio_features',
      'id': '4Z6e88z4yPI3oYBAUicQ1b',
      'uri': 'spotify:track:4Z6e88z4yPI3oYBAUicQ1b',
      'track_href': 'https://api.spotify.com/v1/tracks/4Z6e88z4yPI3oYBAUicQ1b',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4Z6e88z4yPI3oYBAUicQ1b',
      'duration_ms': 184307,
      'time_signature': 4},
     {'danceability': 0.273,
      'energy': 0.53,
      'key': 2,
      'loudness': -12.288,
      'mode': 1,
      'speechiness': 0.0442,
      'acousticness': 0.124,
      'instrumentalness': 0.126,
      'liveness': 0.129,
      'valence': 0.251,
      'tempo': 122.927,
      'type': 'audio_features',
      'id': '36K4sP7kCp83pb5jMAEgii',
      'uri': 'spotify:track:36K4sP7kCp83pb5jMAEgii',
      'track_href': 'https://api.spotify.com/v1/tracks/36K4sP7kCp83pb5jMAEgii',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/36K4sP7kCp83pb5jMAEgii',
      'duration_ms': 255493,
      'time_signature': 4},
     {'danceability': 0.581,
      'energy': 0.867,
      'key': 5,
      'loudness': -4.135,
      'mode': 1,
      'speechiness': 0.0437,
      'acousticness': 0.0206,
      'instrumentalness': 0.00122,
      'liveness': 0.0548,
      'valence': 0.676,
      'tempo': 130.057,
      'type': 'audio_features',
      'id': '6fnkSXzRMtwZolujZ4b4Du',
      'uri': 'spotify:track:6fnkSXzRMtwZolujZ4b4Du',
      'track_href': 'https://api.spotify.com/v1/tracks/6fnkSXzRMtwZolujZ4b4Du',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6fnkSXzRMtwZolujZ4b4Du',
      'duration_ms': 226947,
      'time_signature': 4},
     {'danceability': 0.536,
      'energy': 0.571,
      'key': 0,
      'loudness': -6.879,
      'mode': 1,
      'speechiness': 0.0278,
      'acousticness': 0.355,
      'instrumentalness': 0,
      'liveness': 0.0595,
      'valence': 0.397,
      'tempo': 126.909,
      'type': 'audio_features',
      'id': '4mhF2ALcuLuUotXWV8iboY',
      'uri': 'spotify:track:4mhF2ALcuLuUotXWV8iboY',
      'track_href': 'https://api.spotify.com/v1/tracks/4mhF2ALcuLuUotXWV8iboY',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4mhF2ALcuLuUotXWV8iboY',
      'duration_ms': 280933,
      'time_signature': 4},
     {'danceability': 0.628,
      'energy': 0.237,
      'key': 0,
      'loudness': -14.848,
      'mode': 1,
      'speechiness': 0.0391,
      'acousticness': 0.9,
      'instrumentalness': 0.00936,
      'liveness': 0.131,
      'valence': 0.315,
      'tempo': 128.347,
      'type': 'audio_features',
      'id': '5jOlIxWQRb5f1YqxV2zEE2',
      'uri': 'spotify:track:5jOlIxWQRb5f1YqxV2zEE2',
      'track_href': 'https://api.spotify.com/v1/tracks/5jOlIxWQRb5f1YqxV2zEE2',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5jOlIxWQRb5f1YqxV2zEE2',
      'duration_ms': 127293,
      'time_signature': 4},
     {'danceability': 0.677,
      'energy': 0.378,
      'key': 3,
      'loudness': -5.579,
      'mode': 1,
      'speechiness': 0.033,
      'acousticness': 0.259,
      'instrumentalness': 0.000103,
      'liveness': 0.111,
      'valence': 0.247,
      'tempo': 129.665,
      'type': 'audio_features',
      'id': '2H8RU3mT3V55yH1FfO4pS5',
      'uri': 'spotify:track:2H8RU3mT3V55yH1FfO4pS5',
      'track_href': 'https://api.spotify.com/v1/tracks/2H8RU3mT3V55yH1FfO4pS5',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/2H8RU3mT3V55yH1FfO4pS5',
      'duration_ms': 263053,
      'time_signature': 3},
     {'danceability': 0.724,
      'energy': 0.715,
      'key': 10,
      'loudness': -7.738,
      'mode': 1,
      'speechiness': 0.0504,
      'acousticness': 0.425,
      'instrumentalness': 0,
      'liveness': 0.0746,
      'valence': 0.832,
      'tempo': 104.06,
      'type': 'audio_features',
      'id': '3npbtkmTdUlRxGFxViDD0K',
      'uri': 'spotify:track:3npbtkmTdUlRxGFxViDD0K',
      'track_href': 'https://api.spotify.com/v1/tracks/3npbtkmTdUlRxGFxViDD0K',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3npbtkmTdUlRxGFxViDD0K',
      'duration_ms': 254600,
      'time_signature': 4},
     {'danceability': 0.522,
      'energy': 0.751,
      'key': 0,
      'loudness': -8.404,
      'mode': 1,
      'speechiness': 0.061,
      'acousticness': 0.0547,
      'instrumentalness': 0.0276,
      'liveness': 0.0966,
      'valence': 0.149,
      'tempo': 127.192,
      'type': 'audio_features',
      'id': '6fcDAU3eJkxxyuD6kdPDzX',
      'uri': 'spotify:track:6fcDAU3eJkxxyuD6kdPDzX',
      'track_href': 'https://api.spotify.com/v1/tracks/6fcDAU3eJkxxyuD6kdPDzX',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6fcDAU3eJkxxyuD6kdPDzX',
      'duration_ms': 190533,
      'time_signature': 4},
     {'danceability': 0.685,
      'energy': 0.81,
      'key': 9,
      'loudness': -4.444,
      'mode': 1,
      'speechiness': 0.0901,
      'acousticness': 0.215,
      'instrumentalness': 0,
      'liveness': 0.0835,
      'valence': 0.937,
      'tempo': 87.458,
      'type': 'audio_features',
      'id': '47XpijyXBZ3f1Ro9XxOpOF',
      'uri': 'spotify:track:47XpijyXBZ3f1Ro9XxOpOF',
      'track_href': 'https://api.spotify.com/v1/tracks/47XpijyXBZ3f1Ro9XxOpOF',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/47XpijyXBZ3f1Ro9XxOpOF',
      'duration_ms': 174497,
      'time_signature': 4},
     {'danceability': 0.904,
      'energy': 0.439,
      'key': 4,
      'loudness': -6.786,
      'mode': 1,
      'speechiness': 0.0485,
      'acousticness': 0.63,
      'instrumentalness': 0.000248,
      'liveness': 0.492,
      'valence': 0.944,
      'tempo': 129.78,
      'type': 'audio_features',
      'id': '0JHBJUQpsRF24G9A2YYpUp',
      'uri': 'spotify:track:0JHBJUQpsRF24G9A2YYpUp',
      'track_href': 'https://api.spotify.com/v1/tracks/0JHBJUQpsRF24G9A2YYpUp',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0JHBJUQpsRF24G9A2YYpUp',
      'duration_ms': 250307,
      'time_signature': 4},
     {'danceability': 0.717,
      'energy': 0.862,
      'key': 8,
      'loudness': -4.736,
      'mode': 1,
      'speechiness': 0.054,
      'acousticness': 0.00689,
      'instrumentalness': 0,
      'liveness': 0.321,
      'valence': 0.52,
      'tempo': 130.021,
      'type': 'audio_features',
      'id': '1ULa3GfdMKs0MfRpm6xVlu',
      'uri': 'spotify:track:1ULa3GfdMKs0MfRpm6xVlu',
      'track_href': 'https://api.spotify.com/v1/tracks/1ULa3GfdMKs0MfRpm6xVlu',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1ULa3GfdMKs0MfRpm6xVlu',
      'duration_ms': 224933,
      'time_signature': 4},
     {'danceability': 0.551,
      'energy': 0.804,
      'key': 4,
      'loudness': -3.787,
      'mode': 0,
      'speechiness': 0.0622,
      'acousticness': 0.00596,
      'instrumentalness': 0,
      'liveness': 0.241,
      'valence': 0.209,
      'tempo': 133.048,
      'type': 'audio_features',
      'id': '55ROA2V0eyGQKBf4qs8TfA',
      'uri': 'spotify:track:55ROA2V0eyGQKBf4qs8TfA',
      'track_href': 'https://api.spotify.com/v1/tracks/55ROA2V0eyGQKBf4qs8TfA',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/55ROA2V0eyGQKBf4qs8TfA',
      'duration_ms': 285080,
      'time_signature': 4},
     {'danceability': 0.61,
      'energy': 0.8,
      'key': 0,
      'loudness': -5.158,
      'mode': 1,
      'speechiness': 0.11,
      'acousticness': 0.129,
      'instrumentalness': 0,
      'liveness': 0.118,
      'valence': 0.8,
      'tempo': 89.872,
      'type': 'audio_features',
      'id': '4VLS7iLVVjMLa7XcnbzG1m',
      'uri': 'spotify:track:4VLS7iLVVjMLa7XcnbzG1m',
      'track_href': 'https://api.spotify.com/v1/tracks/4VLS7iLVVjMLa7XcnbzG1m',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/4VLS7iLVVjMLa7XcnbzG1m',
      'duration_ms': 213480,
      'time_signature': 4},
     {'danceability': 0.539,
      'energy': 0.968,
      'key': 11,
      'loudness': -3.051,
      'mode': 0,
      'speechiness': 0.0382,
      'acousticness': 0.0876,
      'instrumentalness': 0.000522,
      'liveness': 0.213,
      'valence': 0.951,
      'tempo': 141.941,
      'type': 'audio_features',
      'id': '5MSGHYOz2HC3PYFnk8CNv8',
      'uri': 'spotify:track:5MSGHYOz2HC3PYFnk8CNv8',
      'track_href': 'https://api.spotify.com/v1/tracks/5MSGHYOz2HC3PYFnk8CNv8',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5MSGHYOz2HC3PYFnk8CNv8',
      'duration_ms': 193573,
      'time_signature': 4},
     {'danceability': 0.481,
      'energy': 0.781,
      'key': 0,
      'loudness': -4.531,
      'mode': 1,
      'speechiness': 0.0408,
      'acousticness': 0.0828,
      'instrumentalness': 0,
      'liveness': 0.397,
      'valence': 0.352,
      'tempo': 143.837,
      'type': 'audio_features',
      'id': '6D9XBfUFCpnVN2tSP2bETW',
      'uri': 'spotify:track:6D9XBfUFCpnVN2tSP2bETW',
      'track_href': 'https://api.spotify.com/v1/tracks/6D9XBfUFCpnVN2tSP2bETW',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/6D9XBfUFCpnVN2tSP2bETW',
      'duration_ms': 216378,
      'time_signature': 4},
     {'danceability': 0.42,
      'energy': 0.802,
      'key': 0,
      'loudness': -9.793,
      'mode': 0,
      'speechiness': 0.0654,
      'acousticness': 0.256,
      'instrumentalness': 0,
      'liveness': 0.21,
      'valence': 0.546,
      'tempo': 101.906,
      'type': 'audio_features',
      'id': '1bpaA5Pn4jlo1cRAOBNQnh',
      'uri': 'spotify:track:1bpaA5Pn4jlo1cRAOBNQnh',
      'track_href': 'https://api.spotify.com/v1/tracks/1bpaA5Pn4jlo1cRAOBNQnh',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1bpaA5Pn4jlo1cRAOBNQnh',
      'duration_ms': 298467,
      'time_signature': 4},
     {'danceability': 0.615,
      'energy': 0.923,
      'key': 1,
      'loudness': -3.379,
      'mode': 1,
      'speechiness': 0.14,
      'acousticness': 0.000188,
      'instrumentalness': 9.66e-05,
      'liveness': 0.166,
      'valence': 0.311,
      'tempo': 160.073,
      'type': 'audio_features',
      'id': '5df2wa0VGO0VzHxYUmDwtl',
      'uri': 'spotify:track:5df2wa0VGO0VzHxYUmDwtl',
      'track_href': 'https://api.spotify.com/v1/tracks/5df2wa0VGO0VzHxYUmDwtl',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5df2wa0VGO0VzHxYUmDwtl',
      'duration_ms': 220547,
      'time_signature': 4},
     {'danceability': 0.541,
      'energy': 0.748,
      'key': 2,
      'loudness': -6.321,
      'mode': 1,
      'speechiness': 0.31,
      'acousticness': 0.00141,
      'instrumentalness': 3.1e-06,
      'liveness': 0.322,
      'valence': 0.351,
      'tempo': 79.616,
      'type': 'audio_features',
      'id': '1rW2nkUbmKD2MC22YrU8cX',
      'uri': 'spotify:track:1rW2nkUbmKD2MC22YrU8cX',
      'track_href': 'https://api.spotify.com/v1/tracks/1rW2nkUbmKD2MC22YrU8cX',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1rW2nkUbmKD2MC22YrU8cX',
      'duration_ms': 196427,
      'time_signature': 4}]





```python
# populate the tracks feature table
def process_features(features_list_dicts):

    features_df = pd.DataFrame(features_list_dicts)
    features_df = features_df.set_index('id')
    return features_df

features_df=process_features(track_features)
features_df
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>analysis_url</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_href</th>
      <th>type</th>
      <th>uri</th>
      <th>valence</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1YGa5zwwbzA9lFGPB3HcLt</th>
      <td>0.053100</td>
      <td>https://api.spotify.com/v1/audio-analysis/1YGa...</td>
      <td>0.839</td>
      <td>267333</td>
      <td>0.791</td>
      <td>0.000485</td>
      <td>8</td>
      <td>0.2010</td>
      <td>-7.771</td>
      <td>1</td>
      <td>0.1160</td>
      <td>129.244</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1YGa5zwwbzA9...</td>
      <td>audio_features</td>
      <td>spotify:track:1YGa5zwwbzA9lFGPB3HcLt</td>
      <td>0.8580</td>
    </tr>
    <tr>
      <th>4gOMf7ak5Ycx9BghTCSTBL</th>
      <td>0.003360</td>
      <td>https://api.spotify.com/v1/audio-analysis/4gOM...</td>
      <td>0.728</td>
      <td>278810</td>
      <td>0.779</td>
      <td>0.000001</td>
      <td>4</td>
      <td>0.3250</td>
      <td>-7.528</td>
      <td>0</td>
      <td>0.1900</td>
      <td>174.046</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4gOMf7ak5Ycx...</td>
      <td>audio_features</td>
      <td>spotify:track:4gOMf7ak5Ycx9BghTCSTBL</td>
      <td>0.8520</td>
    </tr>
    <tr>
      <th>7kl337nuuTTVcXJiQqBgwJ</th>
      <td>0.447000</td>
      <td>https://api.spotify.com/v1/audio-analysis/7kl3...</td>
      <td>0.314</td>
      <td>451160</td>
      <td>0.855</td>
      <td>0.854000</td>
      <td>2</td>
      <td>0.1730</td>
      <td>-7.907</td>
      <td>1</td>
      <td>0.0340</td>
      <td>104.983</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/7kl337nuuTTV...</td>
      <td>audio_features</td>
      <td>spotify:track:7kl337nuuTTVcXJiQqBgwJ</td>
      <td>0.8550</td>
    </tr>
    <tr>
      <th>0LAfANg75hYiV1IAEP3vY6</th>
      <td>0.220000</td>
      <td>https://api.spotify.com/v1/audio-analysis/0LAf...</td>
      <td>0.762</td>
      <td>271907</td>
      <td>0.954</td>
      <td>0.000018</td>
      <td>8</td>
      <td>0.0612</td>
      <td>-4.542</td>
      <td>1</td>
      <td>0.1210</td>
      <td>153.960</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/0LAfANg75hYi...</td>
      <td>audio_features</td>
      <td>spotify:track:0LAfANg75hYiV1IAEP3vY6</td>
      <td>0.9330</td>
    </tr>
    <tr>
      <th>0Hpl422q9VhpQu1RBKlnF1</th>
      <td>0.422000</td>
      <td>https://api.spotify.com/v1/audio-analysis/0Hpl...</td>
      <td>0.633</td>
      <td>509307</td>
      <td>0.834</td>
      <td>0.726000</td>
      <td>11</td>
      <td>0.1720</td>
      <td>-12.959</td>
      <td>1</td>
      <td>0.0631</td>
      <td>130.008</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/0Hpl422q9Vhp...</td>
      <td>audio_features</td>
      <td>spotify:track:0Hpl422q9VhpQu1RBKlnF1</td>
      <td>0.5180</td>
    </tr>
    <tr>
      <th>4uTTsXhygWzSjUxXLHZ4HW</th>
      <td>0.126000</td>
      <td>https://api.spotify.com/v1/audio-analysis/4uTT...</td>
      <td>0.724</td>
      <td>231824</td>
      <td>0.667</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.1220</td>
      <td>-4.806</td>
      <td>0</td>
      <td>0.3850</td>
      <td>143.988</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4uTTsXhygWzS...</td>
      <td>audio_features</td>
      <td>spotify:track:4uTTsXhygWzSjUxXLHZ4HW</td>
      <td>0.1610</td>
    </tr>
    <tr>
      <th>1KDnLoIEPRd4iRYzgvDBzo</th>
      <td>0.221000</td>
      <td>https://api.spotify.com/v1/audio-analysis/1KDn...</td>
      <td>0.617</td>
      <td>259147</td>
      <td>0.493</td>
      <td>0.294000</td>
      <td>0</td>
      <td>0.6960</td>
      <td>-12.779</td>
      <td>1</td>
      <td>0.0495</td>
      <td>119.003</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1KDnLoIEPRd4...</td>
      <td>audio_features</td>
      <td>spotify:track:1KDnLoIEPRd4iRYzgvDBzo</td>
      <td>0.1480</td>
    </tr>
    <tr>
      <th>0j9rNb4IHxLgKdLGZ1sd1I</th>
      <td>0.435000</td>
      <td>https://api.spotify.com/v1/audio-analysis/0j9r...</td>
      <td>0.353</td>
      <td>407814</td>
      <td>0.304</td>
      <td>0.000002</td>
      <td>4</td>
      <td>0.8020</td>
      <td>-9.142</td>
      <td>0</td>
      <td>0.0327</td>
      <td>138.494</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/0j9rNb4IHxLg...</td>
      <td>audio_features</td>
      <td>spotify:track:0j9rNb4IHxLgKdLGZ1sd1I</td>
      <td>0.2330</td>
    </tr>
    <tr>
      <th>1Dfst5fQZYoW8QBfo4mUmn</th>
      <td>0.221000</td>
      <td>https://api.spotify.com/v1/audio-analysis/1Dfs...</td>
      <td>0.633</td>
      <td>205267</td>
      <td>0.286</td>
      <td>0.004150</td>
      <td>2</td>
      <td>0.0879</td>
      <td>-9.703</td>
      <td>0</td>
      <td>0.0270</td>
      <td>129.915</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1Dfst5fQZYoW...</td>
      <td>audio_features</td>
      <td>spotify:track:1Dfst5fQZYoW8QBfo4mUmn</td>
      <td>0.1960</td>
    </tr>
    <tr>
      <th>6ZbiaHwI9x7CIxYGOEmXxd</th>
      <td>0.564000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6Zbi...</td>
      <td>0.569</td>
      <td>198012</td>
      <td>0.789</td>
      <td>0.000000</td>
      <td>11</td>
      <td>0.2940</td>
      <td>-4.607</td>
      <td>1</td>
      <td>0.1230</td>
      <td>160.014</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6ZbiaHwI9x7C...</td>
      <td>audio_features</td>
      <td>spotify:track:6ZbiaHwI9x7CIxYGOEmXxd</td>
      <td>0.6030</td>
    </tr>
    <tr>
      <th>3AUz2xcMnn1CDLDtFRCPeV</th>
      <td>0.407000</td>
      <td>https://api.spotify.com/v1/audio-analysis/3AUz...</td>
      <td>0.726</td>
      <td>214520</td>
      <td>0.718</td>
      <td>0.000266</td>
      <td>7</td>
      <td>0.1200</td>
      <td>-5.192</td>
      <td>0</td>
      <td>0.0510</td>
      <td>123.981</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/3AUz2xcMnn1C...</td>
      <td>audio_features</td>
      <td>spotify:track:3AUz2xcMnn1CDLDtFRCPeV</td>
      <td>0.6700</td>
    </tr>
    <tr>
      <th>7qtAgn9mwxygsPOsUDVRRt</th>
      <td>0.053300</td>
      <td>https://api.spotify.com/v1/audio-analysis/7qtA...</td>
      <td>0.524</td>
      <td>254040</td>
      <td>0.904</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.7760</td>
      <td>-2.071</td>
      <td>1</td>
      <td>0.3980</td>
      <td>161.188</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/7qtAgn9mwxyg...</td>
      <td>audio_features</td>
      <td>spotify:track:7qtAgn9mwxygsPOsUDVRRt</td>
      <td>0.6550</td>
    </tr>
    <tr>
      <th>6X6ctfqSkAaaUWNAEt3J3L</th>
      <td>0.950000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6X6c...</td>
      <td>0.808</td>
      <td>161307</td>
      <td>0.404</td>
      <td>0.790000</td>
      <td>1</td>
      <td>0.1240</td>
      <td>-10.124</td>
      <td>0</td>
      <td>0.0427</td>
      <td>98.023</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6X6ctfqSkAaa...</td>
      <td>audio_features</td>
      <td>spotify:track:6X6ctfqSkAaaUWNAEt3J3L</td>
      <td>0.8400</td>
    </tr>
    <tr>
      <th>0MzO0c9Lr1d4mTUQtGhSJX</th>
      <td>0.562000</td>
      <td>https://api.spotify.com/v1/audio-analysis/0MzO...</td>
      <td>0.665</td>
      <td>353973</td>
      <td>0.426</td>
      <td>0.902000</td>
      <td>2</td>
      <td>0.1000</td>
      <td>-11.557</td>
      <td>1</td>
      <td>0.0308</td>
      <td>98.381</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/0MzO0c9Lr1d4...</td>
      <td>audio_features</td>
      <td>spotify:track:0MzO0c9Lr1d4mTUQtGhSJX</td>
      <td>0.6840</td>
    </tr>
    <tr>
      <th>1FuJsPUcEyAELOp64pc7xw</th>
      <td>0.000037</td>
      <td>https://api.spotify.com/v1/audio-analysis/1FuJ...</td>
      <td>0.413</td>
      <td>372987</td>
      <td>0.959</td>
      <td>0.028700</td>
      <td>6</td>
      <td>0.4640</td>
      <td>-4.007</td>
      <td>0</td>
      <td>0.1580</td>
      <td>114.020</td>
      <td>3</td>
      <td>https://api.spotify.com/v1/tracks/1FuJsPUcEyAE...</td>
      <td>audio_features</td>
      <td>spotify:track:1FuJsPUcEyAELOp64pc7xw</td>
      <td>0.1810</td>
    </tr>
    <tr>
      <th>1yka5XpwTBV951mp2OVYcn</th>
      <td>0.671000</td>
      <td>https://api.spotify.com/v1/audio-analysis/1yka...</td>
      <td>0.601</td>
      <td>221359</td>
      <td>0.496</td>
      <td>0.000004</td>
      <td>5</td>
      <td>0.1320</td>
      <td>-7.622</td>
      <td>0</td>
      <td>0.2300</td>
      <td>177.902</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1yka5XpwTBV9...</td>
      <td>audio_features</td>
      <td>spotify:track:1yka5XpwTBV951mp2OVYcn</td>
      <td>0.2920</td>
    </tr>
    <tr>
      <th>4RBd03PBwN3LNr0er6fkxd</th>
      <td>0.403000</td>
      <td>https://api.spotify.com/v1/audio-analysis/4RBd...</td>
      <td>0.124</td>
      <td>240640</td>
      <td>0.579</td>
      <td>0.888000</td>
      <td>8</td>
      <td>0.2260</td>
      <td>-10.067</td>
      <td>1</td>
      <td>0.0464</td>
      <td>176.431</td>
      <td>3</td>
      <td>https://api.spotify.com/v1/tracks/4RBd03PBwN3L...</td>
      <td>audio_features</td>
      <td>spotify:track:4RBd03PBwN3LNr0er6fkxd</td>
      <td>0.1190</td>
    </tr>
    <tr>
      <th>6GygUjupLLKX273CNzZ4kQ</th>
      <td>0.248000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6Gyg...</td>
      <td>0.793</td>
      <td>80309</td>
      <td>0.412</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.5370</td>
      <td>-10.517</td>
      <td>1</td>
      <td>0.2100</td>
      <td>77.729</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6GygUjupLLKX...</td>
      <td>audio_features</td>
      <td>spotify:track:6GygUjupLLKX273CNzZ4kQ</td>
      <td>0.4070</td>
    </tr>
    <tr>
      <th>060oQAg8NRV2gbBTKYIRPA</th>
      <td>0.960000</td>
      <td>https://api.spotify.com/v1/audio-analysis/060o...</td>
      <td>0.136</td>
      <td>244507</td>
      <td>0.217</td>
      <td>0.000174</td>
      <td>7</td>
      <td>0.9630</td>
      <td>-17.341</td>
      <td>0</td>
      <td>0.0622</td>
      <td>66.749</td>
      <td>3</td>
      <td>https://api.spotify.com/v1/tracks/060oQAg8NRV2...</td>
      <td>audio_features</td>
      <td>spotify:track:060oQAg8NRV2gbBTKYIRPA</td>
      <td>0.1780</td>
    </tr>
    <tr>
      <th>5SzEdjMBb17oURYUXF6iGm</th>
      <td>0.002360</td>
      <td>https://api.spotify.com/v1/audio-analysis/5SzE...</td>
      <td>0.387</td>
      <td>194093</td>
      <td>0.977</td>
      <td>0.006610</td>
      <td>8</td>
      <td>0.3500</td>
      <td>-3.242</td>
      <td>1</td>
      <td>0.1360</td>
      <td>139.905</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/5SzEdjMBb17o...</td>
      <td>audio_features</td>
      <td>spotify:track:5SzEdjMBb17oURYUXF6iGm</td>
      <td>0.3870</td>
    </tr>
    <tr>
      <th>6F2zt7QZkdQKKBul5ATSMD</th>
      <td>0.864000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6F2z...</td>
      <td>0.675</td>
      <td>150517</td>
      <td>0.389</td>
      <td>0.000419</td>
      <td>6</td>
      <td>0.1600</td>
      <td>-11.595</td>
      <td>0</td>
      <td>0.0356</td>
      <td>115.923</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6F2zt7QZkdQK...</td>
      <td>audio_features</td>
      <td>spotify:track:6F2zt7QZkdQKKBul5ATSMD</td>
      <td>0.4150</td>
    </tr>
    <tr>
      <th>4NwGrVIkJavdtdfX03hG0B</th>
      <td>0.207000</td>
      <td>https://api.spotify.com/v1/audio-analysis/4NwG...</td>
      <td>0.345</td>
      <td>169867</td>
      <td>0.786</td>
      <td>0.000000</td>
      <td>7</td>
      <td>0.6740</td>
      <td>-5.632</td>
      <td>1</td>
      <td>0.0329</td>
      <td>172.259</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4NwGrVIkJavd...</td>
      <td>audio_features</td>
      <td>spotify:track:4NwGrVIkJavdtdfX03hG0B</td>
      <td>0.8600</td>
    </tr>
    <tr>
      <th>5KzuAU7zxcP0bq0CPdRRyr</th>
      <td>0.017300</td>
      <td>https://api.spotify.com/v1/audio-analysis/5Kzu...</td>
      <td>0.623</td>
      <td>202840</td>
      <td>0.692</td>
      <td>0.008830</td>
      <td>9</td>
      <td>0.1240</td>
      <td>-7.977</td>
      <td>1</td>
      <td>0.0293</td>
      <td>104.977</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/5KzuAU7zxcP0...</td>
      <td>audio_features</td>
      <td>spotify:track:5KzuAU7zxcP0bq0CPdRRyr</td>
      <td>0.1300</td>
    </tr>
    <tr>
      <th>5XvS3t5O7c9X8cSoIIp3At</th>
      <td>0.183000</td>
      <td>https://api.spotify.com/v1/audio-analysis/5XvS...</td>
      <td>0.513</td>
      <td>260067</td>
      <td>0.840</td>
      <td>0.054600</td>
      <td>2</td>
      <td>0.1060</td>
      <td>-6.070</td>
      <td>1</td>
      <td>0.0321</td>
      <td>95.048</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/5XvS3t5O7c9X...</td>
      <td>audio_features</td>
      <td>spotify:track:5XvS3t5O7c9X8cSoIIp3At</td>
      <td>0.2690</td>
    </tr>
    <tr>
      <th>7IaKUljpLoG5eDfEklNM9x</th>
      <td>0.575000</td>
      <td>https://api.spotify.com/v1/audio-analysis/7IaK...</td>
      <td>0.635</td>
      <td>268547</td>
      <td>0.358</td>
      <td>0.000196</td>
      <td>9</td>
      <td>0.0961</td>
      <td>-10.715</td>
      <td>0</td>
      <td>0.0551</td>
      <td>134.904</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/7IaKUljpLoG5...</td>
      <td>audio_features</td>
      <td>spotify:track:7IaKUljpLoG5eDfEklNM9x</td>
      <td>0.2540</td>
    </tr>
    <tr>
      <th>3g0zYe7PwBeF0iPiYFCSfu</th>
      <td>0.734000</td>
      <td>https://api.spotify.com/v1/audio-analysis/3g0z...</td>
      <td>0.372</td>
      <td>131293</td>
      <td>0.798</td>
      <td>0.000000</td>
      <td>7</td>
      <td>0.9240</td>
      <td>-7.835</td>
      <td>1</td>
      <td>0.0554</td>
      <td>102.722</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/3g0zYe7PwBeF...</td>
      <td>audio_features</td>
      <td>spotify:track:3g0zYe7PwBeF0iPiYFCSfu</td>
      <td>0.8510</td>
    </tr>
    <tr>
      <th>4wPbR6XonWB7fiyWUMAaH2</th>
      <td>0.013200</td>
      <td>https://api.spotify.com/v1/audio-analysis/4wPb...</td>
      <td>0.622</td>
      <td>179307</td>
      <td>0.758</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0615</td>
      <td>-5.384</td>
      <td>1</td>
      <td>0.0603</td>
      <td>119.927</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4wPbR6XonWB7...</td>
      <td>audio_features</td>
      <td>spotify:track:4wPbR6XonWB7fiyWUMAaH2</td>
      <td>0.3270</td>
    </tr>
    <tr>
      <th>2aOmlow495KuwYCcT8ZD4l</th>
      <td>0.642000</td>
      <td>https://api.spotify.com/v1/audio-analysis/2aOm...</td>
      <td>0.515</td>
      <td>203121</td>
      <td>0.397</td>
      <td>0.000003</td>
      <td>0</td>
      <td>0.1460</td>
      <td>-7.661</td>
      <td>1</td>
      <td>0.0280</td>
      <td>77.527</td>
      <td>1</td>
      <td>https://api.spotify.com/v1/tracks/2aOmlow495Ku...</td>
      <td>audio_features</td>
      <td>spotify:track:2aOmlow495KuwYCcT8ZD4l</td>
      <td>0.3650</td>
    </tr>
    <tr>
      <th>6EmP2GA5oTASX7I2VWsHW0</th>
      <td>0.987000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6EmP...</td>
      <td>0.385</td>
      <td>246907</td>
      <td>0.235</td>
      <td>0.898000</td>
      <td>1</td>
      <td>0.1930</td>
      <td>-18.742</td>
      <td>1</td>
      <td>0.0348</td>
      <td>110.048</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6EmP2GA5oTAS...</td>
      <td>audio_features</td>
      <td>spotify:track:6EmP2GA5oTASX7I2VWsHW0</td>
      <td>0.0615</td>
    </tr>
    <tr>
      <th>7zOVwLxNKAG4FNNSKgJUv6</th>
      <td>0.038800</td>
      <td>https://api.spotify.com/v1/audio-analysis/7zOV...</td>
      <td>0.727</td>
      <td>250476</td>
      <td>0.718</td>
      <td>0.026200</td>
      <td>1</td>
      <td>0.3280</td>
      <td>-7.453</td>
      <td>1</td>
      <td>0.2980</td>
      <td>125.975</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/7zOVwLxNKAG4...</td>
      <td>audio_features</td>
      <td>spotify:track:7zOVwLxNKAG4FNNSKgJUv6</td>
      <td>0.1500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6XKRP0dq5PBVJROcKa9NnJ</th>
      <td>0.168000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6XKR...</td>
      <td>0.894</td>
      <td>194973</td>
      <td>0.879</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0.0851</td>
      <td>-5.023</td>
      <td>0</td>
      <td>0.2600</td>
      <td>101.759</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6XKRP0dq5PBV...</td>
      <td>audio_features</td>
      <td>spotify:track:6XKRP0dq5PBVJROcKa9NnJ</td>
      <td>0.8850</td>
    </tr>
    <tr>
      <th>76QWmZ9lR65gGfe03jmmPP</th>
      <td>0.141000</td>
      <td>https://api.spotify.com/v1/audio-analysis/76QW...</td>
      <td>0.667</td>
      <td>265013</td>
      <td>0.808</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.1150</td>
      <td>-4.667</td>
      <td>1</td>
      <td>0.2650</td>
      <td>83.031</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/76QWmZ9lR65g...</td>
      <td>audio_features</td>
      <td>spotify:track:76QWmZ9lR65gGfe03jmmPP</td>
      <td>0.6530</td>
    </tr>
    <tr>
      <th>4pzeuvQ6REsnqhm5OjEhNy</th>
      <td>0.003320</td>
      <td>https://api.spotify.com/v1/audio-analysis/4pze...</td>
      <td>0.708</td>
      <td>293268</td>
      <td>0.538</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0728</td>
      <td>-13.356</td>
      <td>0</td>
      <td>0.0975</td>
      <td>135.953</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4pzeuvQ6REsn...</td>
      <td>audio_features</td>
      <td>spotify:track:4pzeuvQ6REsnqhm5OjEhNy</td>
      <td>0.8440</td>
    </tr>
    <tr>
      <th>1tUiPKYOob0YbMdVRbr79w</th>
      <td>0.006620</td>
      <td>https://api.spotify.com/v1/audio-analysis/1tUi...</td>
      <td>0.520</td>
      <td>247347</td>
      <td>0.843</td>
      <td>0.000001</td>
      <td>10</td>
      <td>0.1520</td>
      <td>-4.711</td>
      <td>0</td>
      <td>0.0900</td>
      <td>129.984</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1tUiPKYOob0Y...</td>
      <td>audio_features</td>
      <td>spotify:track:1tUiPKYOob0YbMdVRbr79w</td>
      <td>0.5210</td>
    </tr>
    <tr>
      <th>6tSQ3Ds8TkjchyvLNgveVy</th>
      <td>0.223000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6tSQ...</td>
      <td>0.580</td>
      <td>175120</td>
      <td>0.666</td>
      <td>0.901000</td>
      <td>11</td>
      <td>0.1140</td>
      <td>-13.372</td>
      <td>1</td>
      <td>0.0350</td>
      <td>136.001</td>
      <td>5</td>
      <td>https://api.spotify.com/v1/tracks/6tSQ3Ds8Tkjc...</td>
      <td>audio_features</td>
      <td>spotify:track:6tSQ3Ds8TkjchyvLNgveVy</td>
      <td>0.1100</td>
    </tr>
    <tr>
      <th>31RYcUDhkqkH1W2xxnzBjY</th>
      <td>0.022300</td>
      <td>https://api.spotify.com/v1/audio-analysis/31RY...</td>
      <td>0.747</td>
      <td>206316</td>
      <td>0.782</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.0902</td>
      <td>-4.326</td>
      <td>1</td>
      <td>0.2830</td>
      <td>113.990</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/31RYcUDhkqkH...</td>
      <td>audio_features</td>
      <td>spotify:track:31RYcUDhkqkH1W2xxnzBjY</td>
      <td>0.5010</td>
    </tr>
    <tr>
      <th>6urBMNarztXhgZ93wItFTQ</th>
      <td>0.338000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6urB...</td>
      <td>0.634</td>
      <td>192610</td>
      <td>0.667</td>
      <td>0.000004</td>
      <td>6</td>
      <td>0.0788</td>
      <td>-8.189</td>
      <td>1</td>
      <td>0.0617</td>
      <td>96.010</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6urBMNarztXh...</td>
      <td>audio_features</td>
      <td>spotify:track:6urBMNarztXhgZ93wItFTQ</td>
      <td>0.4550</td>
    </tr>
    <tr>
      <th>3IjTEp4wLCr0WVyJkvg1kj</th>
      <td>0.242000</td>
      <td>https://api.spotify.com/v1/audio-analysis/3IjT...</td>
      <td>0.507</td>
      <td>271361</td>
      <td>0.420</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.1220</td>
      <td>-7.177</td>
      <td>1</td>
      <td>0.0282</td>
      <td>137.952</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/3IjTEp4wLCr0...</td>
      <td>audio_features</td>
      <td>spotify:track:3IjTEp4wLCr0WVyJkvg1kj</td>
      <td>0.2820</td>
    </tr>
    <tr>
      <th>6SE1opPh1fvqgOSt0leF0d</th>
      <td>0.085800</td>
      <td>https://api.spotify.com/v1/audio-analysis/6SE1...</td>
      <td>0.766</td>
      <td>204957</td>
      <td>0.741</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0991</td>
      <td>-5.642</td>
      <td>1</td>
      <td>0.2130</td>
      <td>110.050</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6SE1opPh1fvq...</td>
      <td>audio_features</td>
      <td>spotify:track:6SE1opPh1fvqgOSt0leF0d</td>
      <td>0.5700</td>
    </tr>
    <tr>
      <th>7Ht9ePi78nxKAinLC0QVe2</th>
      <td>0.047800</td>
      <td>https://api.spotify.com/v1/audio-analysis/7Ht9...</td>
      <td>0.486</td>
      <td>193187</td>
      <td>0.633</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.2610</td>
      <td>-8.640</td>
      <td>1</td>
      <td>0.0380</td>
      <td>127.979</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/7Ht9ePi78nxK...</td>
      <td>audio_features</td>
      <td>spotify:track:7Ht9ePi78nxKAinLC0QVe2</td>
      <td>0.2380</td>
    </tr>
    <tr>
      <th>06JJ9AUDnBRJPTYRCQk3mF</th>
      <td>0.000003</td>
      <td>https://api.spotify.com/v1/audio-analysis/06JJ...</td>
      <td>0.455</td>
      <td>136093</td>
      <td>0.944</td>
      <td>0.885000</td>
      <td>1</td>
      <td>0.1520</td>
      <td>-2.562</td>
      <td>1</td>
      <td>0.0619</td>
      <td>99.985</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/06JJ9AUDnBRJ...</td>
      <td>audio_features</td>
      <td>spotify:track:06JJ9AUDnBRJPTYRCQk3mF</td>
      <td>0.1370</td>
    </tr>
    <tr>
      <th>6OXt9aSIr4DSxSR3Qjrtgp</th>
      <td>0.302000</td>
      <td>https://api.spotify.com/v1/audio-analysis/6OXt...</td>
      <td>0.540</td>
      <td>314853</td>
      <td>0.643</td>
      <td>0.000135</td>
      <td>11</td>
      <td>0.0881</td>
      <td>-9.088</td>
      <td>0</td>
      <td>0.0485</td>
      <td>159.929</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6OXt9aSIr4DS...</td>
      <td>audio_features</td>
      <td>spotify:track:6OXt9aSIr4DSxSR3Qjrtgp</td>
      <td>0.5930</td>
    </tr>
    <tr>
      <th>4Z6e88z4yPI3oYBAUicQ1b</th>
      <td>0.000779</td>
      <td>https://api.spotify.com/v1/audio-analysis/4Z6e...</td>
      <td>0.334</td>
      <td>184307</td>
      <td>0.990</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0744</td>
      <td>-3.961</td>
      <td>1</td>
      <td>0.1870</td>
      <td>145.162</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4Z6e88z4yPI3...</td>
      <td>audio_features</td>
      <td>spotify:track:4Z6e88z4yPI3oYBAUicQ1b</td>
      <td>0.3950</td>
    </tr>
    <tr>
      <th>36K4sP7kCp83pb5jMAEgii</th>
      <td>0.124000</td>
      <td>https://api.spotify.com/v1/audio-analysis/36K4...</td>
      <td>0.273</td>
      <td>255493</td>
      <td>0.530</td>
      <td>0.126000</td>
      <td>2</td>
      <td>0.1290</td>
      <td>-12.288</td>
      <td>1</td>
      <td>0.0442</td>
      <td>122.927</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/36K4sP7kCp83...</td>
      <td>audio_features</td>
      <td>spotify:track:36K4sP7kCp83pb5jMAEgii</td>
      <td>0.2510</td>
    </tr>
    <tr>
      <th>6fnkSXzRMtwZolujZ4b4Du</th>
      <td>0.020600</td>
      <td>https://api.spotify.com/v1/audio-analysis/6fnk...</td>
      <td>0.581</td>
      <td>226947</td>
      <td>0.867</td>
      <td>0.001220</td>
      <td>5</td>
      <td>0.0548</td>
      <td>-4.135</td>
      <td>1</td>
      <td>0.0437</td>
      <td>130.057</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6fnkSXzRMtwZ...</td>
      <td>audio_features</td>
      <td>spotify:track:6fnkSXzRMtwZolujZ4b4Du</td>
      <td>0.6760</td>
    </tr>
    <tr>
      <th>4mhF2ALcuLuUotXWV8iboY</th>
      <td>0.355000</td>
      <td>https://api.spotify.com/v1/audio-analysis/4mhF...</td>
      <td>0.536</td>
      <td>280933</td>
      <td>0.571</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0595</td>
      <td>-6.879</td>
      <td>1</td>
      <td>0.0278</td>
      <td>126.909</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4mhF2ALcuLuU...</td>
      <td>audio_features</td>
      <td>spotify:track:4mhF2ALcuLuUotXWV8iboY</td>
      <td>0.3970</td>
    </tr>
    <tr>
      <th>5jOlIxWQRb5f1YqxV2zEE2</th>
      <td>0.900000</td>
      <td>https://api.spotify.com/v1/audio-analysis/5jOl...</td>
      <td>0.628</td>
      <td>127293</td>
      <td>0.237</td>
      <td>0.009360</td>
      <td>0</td>
      <td>0.1310</td>
      <td>-14.848</td>
      <td>1</td>
      <td>0.0391</td>
      <td>128.347</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/5jOlIxWQRb5f...</td>
      <td>audio_features</td>
      <td>spotify:track:5jOlIxWQRb5f1YqxV2zEE2</td>
      <td>0.3150</td>
    </tr>
    <tr>
      <th>2H8RU3mT3V55yH1FfO4pS5</th>
      <td>0.259000</td>
      <td>https://api.spotify.com/v1/audio-analysis/2H8R...</td>
      <td>0.677</td>
      <td>263053</td>
      <td>0.378</td>
      <td>0.000103</td>
      <td>3</td>
      <td>0.1110</td>
      <td>-5.579</td>
      <td>1</td>
      <td>0.0330</td>
      <td>129.665</td>
      <td>3</td>
      <td>https://api.spotify.com/v1/tracks/2H8RU3mT3V55...</td>
      <td>audio_features</td>
      <td>spotify:track:2H8RU3mT3V55yH1FfO4pS5</td>
      <td>0.2470</td>
    </tr>
    <tr>
      <th>3npbtkmTdUlRxGFxViDD0K</th>
      <td>0.425000</td>
      <td>https://api.spotify.com/v1/audio-analysis/3npb...</td>
      <td>0.724</td>
      <td>254600</td>
      <td>0.715</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.0746</td>
      <td>-7.738</td>
      <td>1</td>
      <td>0.0504</td>
      <td>104.060</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/3npbtkmTdUlR...</td>
      <td>audio_features</td>
      <td>spotify:track:3npbtkmTdUlRxGFxViDD0K</td>
      <td>0.8320</td>
    </tr>
    <tr>
      <th>6fcDAU3eJkxxyuD6kdPDzX</th>
      <td>0.054700</td>
      <td>https://api.spotify.com/v1/audio-analysis/6fcD...</td>
      <td>0.522</td>
      <td>190533</td>
      <td>0.751</td>
      <td>0.027600</td>
      <td>0</td>
      <td>0.0966</td>
      <td>-8.404</td>
      <td>1</td>
      <td>0.0610</td>
      <td>127.192</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6fcDAU3eJkxx...</td>
      <td>audio_features</td>
      <td>spotify:track:6fcDAU3eJkxxyuD6kdPDzX</td>
      <td>0.1490</td>
    </tr>
    <tr>
      <th>47XpijyXBZ3f1Ro9XxOpOF</th>
      <td>0.215000</td>
      <td>https://api.spotify.com/v1/audio-analysis/47Xp...</td>
      <td>0.685</td>
      <td>174497</td>
      <td>0.810</td>
      <td>0.000000</td>
      <td>9</td>
      <td>0.0835</td>
      <td>-4.444</td>
      <td>1</td>
      <td>0.0901</td>
      <td>87.458</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/47XpijyXBZ3f...</td>
      <td>audio_features</td>
      <td>spotify:track:47XpijyXBZ3f1Ro9XxOpOF</td>
      <td>0.9370</td>
    </tr>
    <tr>
      <th>0JHBJUQpsRF24G9A2YYpUp</th>
      <td>0.630000</td>
      <td>https://api.spotify.com/v1/audio-analysis/0JHB...</td>
      <td>0.904</td>
      <td>250307</td>
      <td>0.439</td>
      <td>0.000248</td>
      <td>4</td>
      <td>0.4920</td>
      <td>-6.786</td>
      <td>1</td>
      <td>0.0485</td>
      <td>129.780</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/0JHBJUQpsRF2...</td>
      <td>audio_features</td>
      <td>spotify:track:0JHBJUQpsRF24G9A2YYpUp</td>
      <td>0.9440</td>
    </tr>
    <tr>
      <th>1ULa3GfdMKs0MfRpm6xVlu</th>
      <td>0.006890</td>
      <td>https://api.spotify.com/v1/audio-analysis/1ULa...</td>
      <td>0.717</td>
      <td>224933</td>
      <td>0.862</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.3210</td>
      <td>-4.736</td>
      <td>1</td>
      <td>0.0540</td>
      <td>130.021</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1ULa3GfdMKs0...</td>
      <td>audio_features</td>
      <td>spotify:track:1ULa3GfdMKs0MfRpm6xVlu</td>
      <td>0.5200</td>
    </tr>
    <tr>
      <th>55ROA2V0eyGQKBf4qs8TfA</th>
      <td>0.005960</td>
      <td>https://api.spotify.com/v1/audio-analysis/55RO...</td>
      <td>0.551</td>
      <td>285080</td>
      <td>0.804</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.2410</td>
      <td>-3.787</td>
      <td>0</td>
      <td>0.0622</td>
      <td>133.048</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/55ROA2V0eyGQ...</td>
      <td>audio_features</td>
      <td>spotify:track:55ROA2V0eyGQKBf4qs8TfA</td>
      <td>0.2090</td>
    </tr>
    <tr>
      <th>4VLS7iLVVjMLa7XcnbzG1m</th>
      <td>0.129000</td>
      <td>https://api.spotify.com/v1/audio-analysis/4VLS...</td>
      <td>0.610</td>
      <td>213480</td>
      <td>0.800</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.1180</td>
      <td>-5.158</td>
      <td>1</td>
      <td>0.1100</td>
      <td>89.872</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/4VLS7iLVVjML...</td>
      <td>audio_features</td>
      <td>spotify:track:4VLS7iLVVjMLa7XcnbzG1m</td>
      <td>0.8000</td>
    </tr>
    <tr>
      <th>5MSGHYOz2HC3PYFnk8CNv8</th>
      <td>0.087600</td>
      <td>https://api.spotify.com/v1/audio-analysis/5MSG...</td>
      <td>0.539</td>
      <td>193573</td>
      <td>0.968</td>
      <td>0.000522</td>
      <td>11</td>
      <td>0.2130</td>
      <td>-3.051</td>
      <td>0</td>
      <td>0.0382</td>
      <td>141.941</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/5MSGHYOz2HC3...</td>
      <td>audio_features</td>
      <td>spotify:track:5MSGHYOz2HC3PYFnk8CNv8</td>
      <td>0.9510</td>
    </tr>
    <tr>
      <th>6D9XBfUFCpnVN2tSP2bETW</th>
      <td>0.082800</td>
      <td>https://api.spotify.com/v1/audio-analysis/6D9X...</td>
      <td>0.481</td>
      <td>216378</td>
      <td>0.781</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.3970</td>
      <td>-4.531</td>
      <td>1</td>
      <td>0.0408</td>
      <td>143.837</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/6D9XBfUFCpnV...</td>
      <td>audio_features</td>
      <td>spotify:track:6D9XBfUFCpnVN2tSP2bETW</td>
      <td>0.3520</td>
    </tr>
    <tr>
      <th>1bpaA5Pn4jlo1cRAOBNQnh</th>
      <td>0.256000</td>
      <td>https://api.spotify.com/v1/audio-analysis/1bpa...</td>
      <td>0.420</td>
      <td>298467</td>
      <td>0.802</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.2100</td>
      <td>-9.793</td>
      <td>0</td>
      <td>0.0654</td>
      <td>101.906</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1bpaA5Pn4jlo...</td>
      <td>audio_features</td>
      <td>spotify:track:1bpaA5Pn4jlo1cRAOBNQnh</td>
      <td>0.5460</td>
    </tr>
    <tr>
      <th>5df2wa0VGO0VzHxYUmDwtl</th>
      <td>0.000188</td>
      <td>https://api.spotify.com/v1/audio-analysis/5df2...</td>
      <td>0.615</td>
      <td>220547</td>
      <td>0.923</td>
      <td>0.000097</td>
      <td>1</td>
      <td>0.1660</td>
      <td>-3.379</td>
      <td>1</td>
      <td>0.1400</td>
      <td>160.073</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/5df2wa0VGO0V...</td>
      <td>audio_features</td>
      <td>spotify:track:5df2wa0VGO0VzHxYUmDwtl</td>
      <td>0.3110</td>
    </tr>
    <tr>
      <th>1rW2nkUbmKD2MC22YrU8cX</th>
      <td>0.001410</td>
      <td>https://api.spotify.com/v1/audio-analysis/1rW2...</td>
      <td>0.541</td>
      <td>196427</td>
      <td>0.748</td>
      <td>0.000003</td>
      <td>2</td>
      <td>0.3220</td>
      <td>-6.321</td>
      <td>1</td>
      <td>0.3100</td>
      <td>79.616</td>
      <td>4</td>
      <td>https://api.spotify.com/v1/tracks/1rW2nkUbmKD2...</td>
      <td>audio_features</td>
      <td>spotify:track:1rW2nkUbmKD2MC22YrU8cX</td>
      <td>0.3510</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 17 columns</p>
</div>





```python
features_df_updated = features_df.drop(['analysis_url', 'track_href', 'type', 'uri' ], axis=1)
features_df_updated
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1YGa5zwwbzA9lFGPB3HcLt</th>
      <td>0.053100</td>
      <td>0.839</td>
      <td>267333</td>
      <td>0.791</td>
      <td>0.000485</td>
      <td>8</td>
      <td>0.2010</td>
      <td>-7.771</td>
      <td>1</td>
      <td>0.1160</td>
      <td>129.244</td>
      <td>4</td>
      <td>0.8580</td>
    </tr>
    <tr>
      <th>4gOMf7ak5Ycx9BghTCSTBL</th>
      <td>0.003360</td>
      <td>0.728</td>
      <td>278810</td>
      <td>0.779</td>
      <td>0.000001</td>
      <td>4</td>
      <td>0.3250</td>
      <td>-7.528</td>
      <td>0</td>
      <td>0.1900</td>
      <td>174.046</td>
      <td>4</td>
      <td>0.8520</td>
    </tr>
    <tr>
      <th>7kl337nuuTTVcXJiQqBgwJ</th>
      <td>0.447000</td>
      <td>0.314</td>
      <td>451160</td>
      <td>0.855</td>
      <td>0.854000</td>
      <td>2</td>
      <td>0.1730</td>
      <td>-7.907</td>
      <td>1</td>
      <td>0.0340</td>
      <td>104.983</td>
      <td>4</td>
      <td>0.8550</td>
    </tr>
    <tr>
      <th>0LAfANg75hYiV1IAEP3vY6</th>
      <td>0.220000</td>
      <td>0.762</td>
      <td>271907</td>
      <td>0.954</td>
      <td>0.000018</td>
      <td>8</td>
      <td>0.0612</td>
      <td>-4.542</td>
      <td>1</td>
      <td>0.1210</td>
      <td>153.960</td>
      <td>4</td>
      <td>0.9330</td>
    </tr>
    <tr>
      <th>0Hpl422q9VhpQu1RBKlnF1</th>
      <td>0.422000</td>
      <td>0.633</td>
      <td>509307</td>
      <td>0.834</td>
      <td>0.726000</td>
      <td>11</td>
      <td>0.1720</td>
      <td>-12.959</td>
      <td>1</td>
      <td>0.0631</td>
      <td>130.008</td>
      <td>4</td>
      <td>0.5180</td>
    </tr>
    <tr>
      <th>4uTTsXhygWzSjUxXLHZ4HW</th>
      <td>0.126000</td>
      <td>0.724</td>
      <td>231824</td>
      <td>0.667</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.1220</td>
      <td>-4.806</td>
      <td>0</td>
      <td>0.3850</td>
      <td>143.988</td>
      <td>4</td>
      <td>0.1610</td>
    </tr>
    <tr>
      <th>1KDnLoIEPRd4iRYzgvDBzo</th>
      <td>0.221000</td>
      <td>0.617</td>
      <td>259147</td>
      <td>0.493</td>
      <td>0.294000</td>
      <td>0</td>
      <td>0.6960</td>
      <td>-12.779</td>
      <td>1</td>
      <td>0.0495</td>
      <td>119.003</td>
      <td>4</td>
      <td>0.1480</td>
    </tr>
    <tr>
      <th>0j9rNb4IHxLgKdLGZ1sd1I</th>
      <td>0.435000</td>
      <td>0.353</td>
      <td>407814</td>
      <td>0.304</td>
      <td>0.000002</td>
      <td>4</td>
      <td>0.8020</td>
      <td>-9.142</td>
      <td>0</td>
      <td>0.0327</td>
      <td>138.494</td>
      <td>4</td>
      <td>0.2330</td>
    </tr>
    <tr>
      <th>1Dfst5fQZYoW8QBfo4mUmn</th>
      <td>0.221000</td>
      <td>0.633</td>
      <td>205267</td>
      <td>0.286</td>
      <td>0.004150</td>
      <td>2</td>
      <td>0.0879</td>
      <td>-9.703</td>
      <td>0</td>
      <td>0.0270</td>
      <td>129.915</td>
      <td>4</td>
      <td>0.1960</td>
    </tr>
    <tr>
      <th>6ZbiaHwI9x7CIxYGOEmXxd</th>
      <td>0.564000</td>
      <td>0.569</td>
      <td>198012</td>
      <td>0.789</td>
      <td>0.000000</td>
      <td>11</td>
      <td>0.2940</td>
      <td>-4.607</td>
      <td>1</td>
      <td>0.1230</td>
      <td>160.014</td>
      <td>4</td>
      <td>0.6030</td>
    </tr>
    <tr>
      <th>3AUz2xcMnn1CDLDtFRCPeV</th>
      <td>0.407000</td>
      <td>0.726</td>
      <td>214520</td>
      <td>0.718</td>
      <td>0.000266</td>
      <td>7</td>
      <td>0.1200</td>
      <td>-5.192</td>
      <td>0</td>
      <td>0.0510</td>
      <td>123.981</td>
      <td>4</td>
      <td>0.6700</td>
    </tr>
    <tr>
      <th>7qtAgn9mwxygsPOsUDVRRt</th>
      <td>0.053300</td>
      <td>0.524</td>
      <td>254040</td>
      <td>0.904</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.7760</td>
      <td>-2.071</td>
      <td>1</td>
      <td>0.3980</td>
      <td>161.188</td>
      <td>4</td>
      <td>0.6550</td>
    </tr>
    <tr>
      <th>6X6ctfqSkAaaUWNAEt3J3L</th>
      <td>0.950000</td>
      <td>0.808</td>
      <td>161307</td>
      <td>0.404</td>
      <td>0.790000</td>
      <td>1</td>
      <td>0.1240</td>
      <td>-10.124</td>
      <td>0</td>
      <td>0.0427</td>
      <td>98.023</td>
      <td>4</td>
      <td>0.8400</td>
    </tr>
    <tr>
      <th>0MzO0c9Lr1d4mTUQtGhSJX</th>
      <td>0.562000</td>
      <td>0.665</td>
      <td>353973</td>
      <td>0.426</td>
      <td>0.902000</td>
      <td>2</td>
      <td>0.1000</td>
      <td>-11.557</td>
      <td>1</td>
      <td>0.0308</td>
      <td>98.381</td>
      <td>4</td>
      <td>0.6840</td>
    </tr>
    <tr>
      <th>1FuJsPUcEyAELOp64pc7xw</th>
      <td>0.000037</td>
      <td>0.413</td>
      <td>372987</td>
      <td>0.959</td>
      <td>0.028700</td>
      <td>6</td>
      <td>0.4640</td>
      <td>-4.007</td>
      <td>0</td>
      <td>0.1580</td>
      <td>114.020</td>
      <td>3</td>
      <td>0.1810</td>
    </tr>
    <tr>
      <th>1yka5XpwTBV951mp2OVYcn</th>
      <td>0.671000</td>
      <td>0.601</td>
      <td>221359</td>
      <td>0.496</td>
      <td>0.000004</td>
      <td>5</td>
      <td>0.1320</td>
      <td>-7.622</td>
      <td>0</td>
      <td>0.2300</td>
      <td>177.902</td>
      <td>4</td>
      <td>0.2920</td>
    </tr>
    <tr>
      <th>4RBd03PBwN3LNr0er6fkxd</th>
      <td>0.403000</td>
      <td>0.124</td>
      <td>240640</td>
      <td>0.579</td>
      <td>0.888000</td>
      <td>8</td>
      <td>0.2260</td>
      <td>-10.067</td>
      <td>1</td>
      <td>0.0464</td>
      <td>176.431</td>
      <td>3</td>
      <td>0.1190</td>
    </tr>
    <tr>
      <th>6GygUjupLLKX273CNzZ4kQ</th>
      <td>0.248000</td>
      <td>0.793</td>
      <td>80309</td>
      <td>0.412</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.5370</td>
      <td>-10.517</td>
      <td>1</td>
      <td>0.2100</td>
      <td>77.729</td>
      <td>4</td>
      <td>0.4070</td>
    </tr>
    <tr>
      <th>060oQAg8NRV2gbBTKYIRPA</th>
      <td>0.960000</td>
      <td>0.136</td>
      <td>244507</td>
      <td>0.217</td>
      <td>0.000174</td>
      <td>7</td>
      <td>0.9630</td>
      <td>-17.341</td>
      <td>0</td>
      <td>0.0622</td>
      <td>66.749</td>
      <td>3</td>
      <td>0.1780</td>
    </tr>
    <tr>
      <th>5SzEdjMBb17oURYUXF6iGm</th>
      <td>0.002360</td>
      <td>0.387</td>
      <td>194093</td>
      <td>0.977</td>
      <td>0.006610</td>
      <td>8</td>
      <td>0.3500</td>
      <td>-3.242</td>
      <td>1</td>
      <td>0.1360</td>
      <td>139.905</td>
      <td>4</td>
      <td>0.3870</td>
    </tr>
    <tr>
      <th>6F2zt7QZkdQKKBul5ATSMD</th>
      <td>0.864000</td>
      <td>0.675</td>
      <td>150517</td>
      <td>0.389</td>
      <td>0.000419</td>
      <td>6</td>
      <td>0.1600</td>
      <td>-11.595</td>
      <td>0</td>
      <td>0.0356</td>
      <td>115.923</td>
      <td>4</td>
      <td>0.4150</td>
    </tr>
    <tr>
      <th>4NwGrVIkJavdtdfX03hG0B</th>
      <td>0.207000</td>
      <td>0.345</td>
      <td>169867</td>
      <td>0.786</td>
      <td>0.000000</td>
      <td>7</td>
      <td>0.6740</td>
      <td>-5.632</td>
      <td>1</td>
      <td>0.0329</td>
      <td>172.259</td>
      <td>4</td>
      <td>0.8600</td>
    </tr>
    <tr>
      <th>5KzuAU7zxcP0bq0CPdRRyr</th>
      <td>0.017300</td>
      <td>0.623</td>
      <td>202840</td>
      <td>0.692</td>
      <td>0.008830</td>
      <td>9</td>
      <td>0.1240</td>
      <td>-7.977</td>
      <td>1</td>
      <td>0.0293</td>
      <td>104.977</td>
      <td>4</td>
      <td>0.1300</td>
    </tr>
    <tr>
      <th>5XvS3t5O7c9X8cSoIIp3At</th>
      <td>0.183000</td>
      <td>0.513</td>
      <td>260067</td>
      <td>0.840</td>
      <td>0.054600</td>
      <td>2</td>
      <td>0.1060</td>
      <td>-6.070</td>
      <td>1</td>
      <td>0.0321</td>
      <td>95.048</td>
      <td>4</td>
      <td>0.2690</td>
    </tr>
    <tr>
      <th>7IaKUljpLoG5eDfEklNM9x</th>
      <td>0.575000</td>
      <td>0.635</td>
      <td>268547</td>
      <td>0.358</td>
      <td>0.000196</td>
      <td>9</td>
      <td>0.0961</td>
      <td>-10.715</td>
      <td>0</td>
      <td>0.0551</td>
      <td>134.904</td>
      <td>4</td>
      <td>0.2540</td>
    </tr>
    <tr>
      <th>3g0zYe7PwBeF0iPiYFCSfu</th>
      <td>0.734000</td>
      <td>0.372</td>
      <td>131293</td>
      <td>0.798</td>
      <td>0.000000</td>
      <td>7</td>
      <td>0.9240</td>
      <td>-7.835</td>
      <td>1</td>
      <td>0.0554</td>
      <td>102.722</td>
      <td>4</td>
      <td>0.8510</td>
    </tr>
    <tr>
      <th>4wPbR6XonWB7fiyWUMAaH2</th>
      <td>0.013200</td>
      <td>0.622</td>
      <td>179307</td>
      <td>0.758</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0615</td>
      <td>-5.384</td>
      <td>1</td>
      <td>0.0603</td>
      <td>119.927</td>
      <td>4</td>
      <td>0.3270</td>
    </tr>
    <tr>
      <th>2aOmlow495KuwYCcT8ZD4l</th>
      <td>0.642000</td>
      <td>0.515</td>
      <td>203121</td>
      <td>0.397</td>
      <td>0.000003</td>
      <td>0</td>
      <td>0.1460</td>
      <td>-7.661</td>
      <td>1</td>
      <td>0.0280</td>
      <td>77.527</td>
      <td>1</td>
      <td>0.3650</td>
    </tr>
    <tr>
      <th>6EmP2GA5oTASX7I2VWsHW0</th>
      <td>0.987000</td>
      <td>0.385</td>
      <td>246907</td>
      <td>0.235</td>
      <td>0.898000</td>
      <td>1</td>
      <td>0.1930</td>
      <td>-18.742</td>
      <td>1</td>
      <td>0.0348</td>
      <td>110.048</td>
      <td>4</td>
      <td>0.0615</td>
    </tr>
    <tr>
      <th>7zOVwLxNKAG4FNNSKgJUv6</th>
      <td>0.038800</td>
      <td>0.727</td>
      <td>250476</td>
      <td>0.718</td>
      <td>0.026200</td>
      <td>1</td>
      <td>0.3280</td>
      <td>-7.453</td>
      <td>1</td>
      <td>0.2980</td>
      <td>125.975</td>
      <td>4</td>
      <td>0.1500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6XKRP0dq5PBVJROcKa9NnJ</th>
      <td>0.168000</td>
      <td>0.894</td>
      <td>194973</td>
      <td>0.879</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0.0851</td>
      <td>-5.023</td>
      <td>0</td>
      <td>0.2600</td>
      <td>101.759</td>
      <td>4</td>
      <td>0.8850</td>
    </tr>
    <tr>
      <th>76QWmZ9lR65gGfe03jmmPP</th>
      <td>0.141000</td>
      <td>0.667</td>
      <td>265013</td>
      <td>0.808</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.1150</td>
      <td>-4.667</td>
      <td>1</td>
      <td>0.2650</td>
      <td>83.031</td>
      <td>4</td>
      <td>0.6530</td>
    </tr>
    <tr>
      <th>4pzeuvQ6REsnqhm5OjEhNy</th>
      <td>0.003320</td>
      <td>0.708</td>
      <td>293268</td>
      <td>0.538</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0728</td>
      <td>-13.356</td>
      <td>0</td>
      <td>0.0975</td>
      <td>135.953</td>
      <td>4</td>
      <td>0.8440</td>
    </tr>
    <tr>
      <th>1tUiPKYOob0YbMdVRbr79w</th>
      <td>0.006620</td>
      <td>0.520</td>
      <td>247347</td>
      <td>0.843</td>
      <td>0.000001</td>
      <td>10</td>
      <td>0.1520</td>
      <td>-4.711</td>
      <td>0</td>
      <td>0.0900</td>
      <td>129.984</td>
      <td>4</td>
      <td>0.5210</td>
    </tr>
    <tr>
      <th>6tSQ3Ds8TkjchyvLNgveVy</th>
      <td>0.223000</td>
      <td>0.580</td>
      <td>175120</td>
      <td>0.666</td>
      <td>0.901000</td>
      <td>11</td>
      <td>0.1140</td>
      <td>-13.372</td>
      <td>1</td>
      <td>0.0350</td>
      <td>136.001</td>
      <td>5</td>
      <td>0.1100</td>
    </tr>
    <tr>
      <th>31RYcUDhkqkH1W2xxnzBjY</th>
      <td>0.022300</td>
      <td>0.747</td>
      <td>206316</td>
      <td>0.782</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.0902</td>
      <td>-4.326</td>
      <td>1</td>
      <td>0.2830</td>
      <td>113.990</td>
      <td>4</td>
      <td>0.5010</td>
    </tr>
    <tr>
      <th>6urBMNarztXhgZ93wItFTQ</th>
      <td>0.338000</td>
      <td>0.634</td>
      <td>192610</td>
      <td>0.667</td>
      <td>0.000004</td>
      <td>6</td>
      <td>0.0788</td>
      <td>-8.189</td>
      <td>1</td>
      <td>0.0617</td>
      <td>96.010</td>
      <td>4</td>
      <td>0.4550</td>
    </tr>
    <tr>
      <th>3IjTEp4wLCr0WVyJkvg1kj</th>
      <td>0.242000</td>
      <td>0.507</td>
      <td>271361</td>
      <td>0.420</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.1220</td>
      <td>-7.177</td>
      <td>1</td>
      <td>0.0282</td>
      <td>137.952</td>
      <td>4</td>
      <td>0.2820</td>
    </tr>
    <tr>
      <th>6SE1opPh1fvqgOSt0leF0d</th>
      <td>0.085800</td>
      <td>0.766</td>
      <td>204957</td>
      <td>0.741</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0991</td>
      <td>-5.642</td>
      <td>1</td>
      <td>0.2130</td>
      <td>110.050</td>
      <td>4</td>
      <td>0.5700</td>
    </tr>
    <tr>
      <th>7Ht9ePi78nxKAinLC0QVe2</th>
      <td>0.047800</td>
      <td>0.486</td>
      <td>193187</td>
      <td>0.633</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.2610</td>
      <td>-8.640</td>
      <td>1</td>
      <td>0.0380</td>
      <td>127.979</td>
      <td>4</td>
      <td>0.2380</td>
    </tr>
    <tr>
      <th>06JJ9AUDnBRJPTYRCQk3mF</th>
      <td>0.000003</td>
      <td>0.455</td>
      <td>136093</td>
      <td>0.944</td>
      <td>0.885000</td>
      <td>1</td>
      <td>0.1520</td>
      <td>-2.562</td>
      <td>1</td>
      <td>0.0619</td>
      <td>99.985</td>
      <td>4</td>
      <td>0.1370</td>
    </tr>
    <tr>
      <th>6OXt9aSIr4DSxSR3Qjrtgp</th>
      <td>0.302000</td>
      <td>0.540</td>
      <td>314853</td>
      <td>0.643</td>
      <td>0.000135</td>
      <td>11</td>
      <td>0.0881</td>
      <td>-9.088</td>
      <td>0</td>
      <td>0.0485</td>
      <td>159.929</td>
      <td>4</td>
      <td>0.5930</td>
    </tr>
    <tr>
      <th>4Z6e88z4yPI3oYBAUicQ1b</th>
      <td>0.000779</td>
      <td>0.334</td>
      <td>184307</td>
      <td>0.990</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0744</td>
      <td>-3.961</td>
      <td>1</td>
      <td>0.1870</td>
      <td>145.162</td>
      <td>4</td>
      <td>0.3950</td>
    </tr>
    <tr>
      <th>36K4sP7kCp83pb5jMAEgii</th>
      <td>0.124000</td>
      <td>0.273</td>
      <td>255493</td>
      <td>0.530</td>
      <td>0.126000</td>
      <td>2</td>
      <td>0.1290</td>
      <td>-12.288</td>
      <td>1</td>
      <td>0.0442</td>
      <td>122.927</td>
      <td>4</td>
      <td>0.2510</td>
    </tr>
    <tr>
      <th>6fnkSXzRMtwZolujZ4b4Du</th>
      <td>0.020600</td>
      <td>0.581</td>
      <td>226947</td>
      <td>0.867</td>
      <td>0.001220</td>
      <td>5</td>
      <td>0.0548</td>
      <td>-4.135</td>
      <td>1</td>
      <td>0.0437</td>
      <td>130.057</td>
      <td>4</td>
      <td>0.6760</td>
    </tr>
    <tr>
      <th>4mhF2ALcuLuUotXWV8iboY</th>
      <td>0.355000</td>
      <td>0.536</td>
      <td>280933</td>
      <td>0.571</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0595</td>
      <td>-6.879</td>
      <td>1</td>
      <td>0.0278</td>
      <td>126.909</td>
      <td>4</td>
      <td>0.3970</td>
    </tr>
    <tr>
      <th>5jOlIxWQRb5f1YqxV2zEE2</th>
      <td>0.900000</td>
      <td>0.628</td>
      <td>127293</td>
      <td>0.237</td>
      <td>0.009360</td>
      <td>0</td>
      <td>0.1310</td>
      <td>-14.848</td>
      <td>1</td>
      <td>0.0391</td>
      <td>128.347</td>
      <td>4</td>
      <td>0.3150</td>
    </tr>
    <tr>
      <th>2H8RU3mT3V55yH1FfO4pS5</th>
      <td>0.259000</td>
      <td>0.677</td>
      <td>263053</td>
      <td>0.378</td>
      <td>0.000103</td>
      <td>3</td>
      <td>0.1110</td>
      <td>-5.579</td>
      <td>1</td>
      <td>0.0330</td>
      <td>129.665</td>
      <td>3</td>
      <td>0.2470</td>
    </tr>
    <tr>
      <th>3npbtkmTdUlRxGFxViDD0K</th>
      <td>0.425000</td>
      <td>0.724</td>
      <td>254600</td>
      <td>0.715</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.0746</td>
      <td>-7.738</td>
      <td>1</td>
      <td>0.0504</td>
      <td>104.060</td>
      <td>4</td>
      <td>0.8320</td>
    </tr>
    <tr>
      <th>6fcDAU3eJkxxyuD6kdPDzX</th>
      <td>0.054700</td>
      <td>0.522</td>
      <td>190533</td>
      <td>0.751</td>
      <td>0.027600</td>
      <td>0</td>
      <td>0.0966</td>
      <td>-8.404</td>
      <td>1</td>
      <td>0.0610</td>
      <td>127.192</td>
      <td>4</td>
      <td>0.1490</td>
    </tr>
    <tr>
      <th>47XpijyXBZ3f1Ro9XxOpOF</th>
      <td>0.215000</td>
      <td>0.685</td>
      <td>174497</td>
      <td>0.810</td>
      <td>0.000000</td>
      <td>9</td>
      <td>0.0835</td>
      <td>-4.444</td>
      <td>1</td>
      <td>0.0901</td>
      <td>87.458</td>
      <td>4</td>
      <td>0.9370</td>
    </tr>
    <tr>
      <th>0JHBJUQpsRF24G9A2YYpUp</th>
      <td>0.630000</td>
      <td>0.904</td>
      <td>250307</td>
      <td>0.439</td>
      <td>0.000248</td>
      <td>4</td>
      <td>0.4920</td>
      <td>-6.786</td>
      <td>1</td>
      <td>0.0485</td>
      <td>129.780</td>
      <td>4</td>
      <td>0.9440</td>
    </tr>
    <tr>
      <th>1ULa3GfdMKs0MfRpm6xVlu</th>
      <td>0.006890</td>
      <td>0.717</td>
      <td>224933</td>
      <td>0.862</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.3210</td>
      <td>-4.736</td>
      <td>1</td>
      <td>0.0540</td>
      <td>130.021</td>
      <td>4</td>
      <td>0.5200</td>
    </tr>
    <tr>
      <th>55ROA2V0eyGQKBf4qs8TfA</th>
      <td>0.005960</td>
      <td>0.551</td>
      <td>285080</td>
      <td>0.804</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.2410</td>
      <td>-3.787</td>
      <td>0</td>
      <td>0.0622</td>
      <td>133.048</td>
      <td>4</td>
      <td>0.2090</td>
    </tr>
    <tr>
      <th>4VLS7iLVVjMLa7XcnbzG1m</th>
      <td>0.129000</td>
      <td>0.610</td>
      <td>213480</td>
      <td>0.800</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.1180</td>
      <td>-5.158</td>
      <td>1</td>
      <td>0.1100</td>
      <td>89.872</td>
      <td>4</td>
      <td>0.8000</td>
    </tr>
    <tr>
      <th>5MSGHYOz2HC3PYFnk8CNv8</th>
      <td>0.087600</td>
      <td>0.539</td>
      <td>193573</td>
      <td>0.968</td>
      <td>0.000522</td>
      <td>11</td>
      <td>0.2130</td>
      <td>-3.051</td>
      <td>0</td>
      <td>0.0382</td>
      <td>141.941</td>
      <td>4</td>
      <td>0.9510</td>
    </tr>
    <tr>
      <th>6D9XBfUFCpnVN2tSP2bETW</th>
      <td>0.082800</td>
      <td>0.481</td>
      <td>216378</td>
      <td>0.781</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.3970</td>
      <td>-4.531</td>
      <td>1</td>
      <td>0.0408</td>
      <td>143.837</td>
      <td>4</td>
      <td>0.3520</td>
    </tr>
    <tr>
      <th>1bpaA5Pn4jlo1cRAOBNQnh</th>
      <td>0.256000</td>
      <td>0.420</td>
      <td>298467</td>
      <td>0.802</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.2100</td>
      <td>-9.793</td>
      <td>0</td>
      <td>0.0654</td>
      <td>101.906</td>
      <td>4</td>
      <td>0.5460</td>
    </tr>
    <tr>
      <th>5df2wa0VGO0VzHxYUmDwtl</th>
      <td>0.000188</td>
      <td>0.615</td>
      <td>220547</td>
      <td>0.923</td>
      <td>0.000097</td>
      <td>1</td>
      <td>0.1660</td>
      <td>-3.379</td>
      <td>1</td>
      <td>0.1400</td>
      <td>160.073</td>
      <td>4</td>
      <td>0.3110</td>
    </tr>
    <tr>
      <th>1rW2nkUbmKD2MC22YrU8cX</th>
      <td>0.001410</td>
      <td>0.541</td>
      <td>196427</td>
      <td>0.748</td>
      <td>0.000003</td>
      <td>2</td>
      <td>0.3220</td>
      <td>-6.321</td>
      <td>1</td>
      <td>0.3100</td>
      <td>79.616</td>
      <td>4</td>
      <td>0.3510</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 13 columns</p>
</div>





```python
# create the tracks feature table
query = 'DROP TABLE IF EXISTS track_features;'
c.execute(query)
conn.commit()

c.execute('''CREATE TABLE IF NOT EXISTS track_features (id varchar(255) PRIMARY KEY,
         acousticness integer,danceability integer, duration_ms integer, energy integer,
         instrumentalness integer, key integer, liveness integer, loudness integer,
         mode integer, speechiness integer, tempo integer, time_signature integer,
         valence integer
         );''')
conn.commit()
```




```python
# subselect 1000 tracks from the X pickle file
# tracks_path_sub = X[:100].copy()
# len(tracks_path_sub)

# prelocate list to host processed and unprocessed files
# processed_files = []
# unprocessed_files = X.copy()

# # loop over file subsets
# for file_ind, filepath in enumerate(tracks_path_sub):
#     # keep track of files that have been processed
#     print('File number = ', file_ind)
#     unprocessed_files.remove(filepath)
#     processed_files.append(filepath)

#     # load the file
#     with open(filepath, "r") as fd:
#         data = json.load(fd)

#     # get tracks dataframe for insertion
#     tracks_df_to_insert = prepare_tracks_update(tracks_df, playlist_id, conn, cur)

#     # insert tracks dataframe into tracks table
#     tracks_df_to_insert.to_sql('tracks', conn, if_exists='append')
```




```python
# insert track features dataframe into track features SQL table
features_df_updated.to_sql('track_features', conn, if_exists='append')
```




```python
# read the track features table
pd.read_sql_query("select * from track_features LIMIT 10;", conn)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1YGa5zwwbzA9lFGPB3HcLt</td>
      <td>0.05310</td>
      <td>0.839</td>
      <td>267333</td>
      <td>0.791</td>
      <td>0.000485</td>
      <td>8</td>
      <td>0.2010</td>
      <td>-7.771</td>
      <td>1</td>
      <td>0.1160</td>
      <td>129.244</td>
      <td>4</td>
      <td>0.858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4gOMf7ak5Ycx9BghTCSTBL</td>
      <td>0.00336</td>
      <td>0.728</td>
      <td>278810</td>
      <td>0.779</td>
      <td>0.000001</td>
      <td>4</td>
      <td>0.3250</td>
      <td>-7.528</td>
      <td>0</td>
      <td>0.1900</td>
      <td>174.046</td>
      <td>4</td>
      <td>0.852</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7kl337nuuTTVcXJiQqBgwJ</td>
      <td>0.44700</td>
      <td>0.314</td>
      <td>451160</td>
      <td>0.855</td>
      <td>0.854000</td>
      <td>2</td>
      <td>0.1730</td>
      <td>-7.907</td>
      <td>1</td>
      <td>0.0340</td>
      <td>104.983</td>
      <td>4</td>
      <td>0.855</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0LAfANg75hYiV1IAEP3vY6</td>
      <td>0.22000</td>
      <td>0.762</td>
      <td>271907</td>
      <td>0.954</td>
      <td>0.000018</td>
      <td>8</td>
      <td>0.0612</td>
      <td>-4.542</td>
      <td>1</td>
      <td>0.1210</td>
      <td>153.960</td>
      <td>4</td>
      <td>0.933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0Hpl422q9VhpQu1RBKlnF1</td>
      <td>0.42200</td>
      <td>0.633</td>
      <td>509307</td>
      <td>0.834</td>
      <td>0.726000</td>
      <td>11</td>
      <td>0.1720</td>
      <td>-12.959</td>
      <td>1</td>
      <td>0.0631</td>
      <td>130.008</td>
      <td>4</td>
      <td>0.518</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4uTTsXhygWzSjUxXLHZ4HW</td>
      <td>0.12600</td>
      <td>0.724</td>
      <td>231824</td>
      <td>0.667</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.1220</td>
      <td>-4.806</td>
      <td>0</td>
      <td>0.3850</td>
      <td>143.988</td>
      <td>4</td>
      <td>0.161</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1KDnLoIEPRd4iRYzgvDBzo</td>
      <td>0.22100</td>
      <td>0.617</td>
      <td>259147</td>
      <td>0.493</td>
      <td>0.294000</td>
      <td>0</td>
      <td>0.6960</td>
      <td>-12.779</td>
      <td>1</td>
      <td>0.0495</td>
      <td>119.003</td>
      <td>4</td>
      <td>0.148</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0j9rNb4IHxLgKdLGZ1sd1I</td>
      <td>0.43500</td>
      <td>0.353</td>
      <td>407814</td>
      <td>0.304</td>
      <td>0.000002</td>
      <td>4</td>
      <td>0.8020</td>
      <td>-9.142</td>
      <td>0</td>
      <td>0.0327</td>
      <td>138.494</td>
      <td>4</td>
      <td>0.233</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1Dfst5fQZYoW8QBfo4mUmn</td>
      <td>0.22100</td>
      <td>0.633</td>
      <td>205267</td>
      <td>0.286</td>
      <td>0.004150</td>
      <td>2</td>
      <td>0.0879</td>
      <td>-9.703</td>
      <td>0</td>
      <td>0.0270</td>
      <td>129.915</td>
      <td>4</td>
      <td>0.196</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6ZbiaHwI9x7CIxYGOEmXxd</td>
      <td>0.56400</td>
      <td>0.569</td>
      <td>198012</td>
      <td>0.789</td>
      <td>0.000000</td>
      <td>11</td>
      <td>0.2940</td>
      <td>-4.607</td>
      <td>1</td>
      <td>0.1230</td>
      <td>160.014</td>
      <td>4</td>
      <td>0.603</td>
    </tr>
  </tbody>
</table>
</div>





```python
# read the table
pd.read_sql_query("select * from tracks LIMIT 10;", conn)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>track_name</th>
      <th>album</th>
      <th>album_name</th>
      <th>artist</th>
      <th>artist_name</th>
      <th>duration_ms</th>
      <th>playlist_member</th>
      <th>num_member</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1Pm3fq1SC6lUlNVBGZi3Em</td>
      <td>Concerning Hobbits (The Lord of the Rings)</td>
      <td>3q8vR3PFV8kG1m1Iv8DpKq</td>
      <td>Versus Hollywood</td>
      <td>7zdmbPudNX4SQJXnYIuCTC</td>
      <td>Daniel Tidwell</td>
      <td>108532</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6BAwTzs5CWWuaaOpp6mTeL</td>
      <td>End It Now!</td>
      <td>4SKnp9LlgXiEeVVud74fTS</td>
      <td>Arrows</td>
      <td>4Xo1N4V7w6qX23OHAPcLCe</td>
      <td>The Lonely Forest</td>
      <td>261146</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3Ex4oMC0UeAQNAG78AuVfi</td>
      <td>Pink City</td>
      <td>4ukM0IdJpt2NL8F8kDebS7</td>
      <td>A Promise</td>
      <td>5JLqvjW3Nyom2OsRUyFsS9</td>
      <td>Xiu Xiu</td>
      <td>133373</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17q3q8wMWgYB8HoqZNNMRT</td>
      <td>So Much Love So Little Time</td>
      <td>3jlchQb5BMfS7Cug5iceJn</td>
      <td>Live A Little Love A Lot</td>
      <td>5juac7bFYyLKmV0VGSyaKM</td>
      <td>Moose</td>
      <td>202533</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6AAWJJ0zamfEVReLLDj4BC</td>
      <td>Screaming</td>
      <td>5xxnHsOdealeAwAjfJfWH6</td>
      <td>. . . XYZ</td>
      <td>5juac7bFYyLKmV0VGSyaKM</td>
      <td>Moose</td>
      <td>233733</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1KVU56xCG8SqaXRjf9n4Da</td>
      <td>Crush The Camera</td>
      <td>0ZvFi6ooJJOuMuMWfHAstk</td>
      <td>10:1</td>
      <td>2JSc53B5cQ31m0xTB7JFpG</td>
      <td>Rogue Wave</td>
      <td>180666</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7331X4zR3C2eaLh5iHN98k</td>
      <td>Interference</td>
      <td>5mmZbh9XqQaR0AEeevo5TS</td>
      <td>Interference / Smile &amp; Gesture</td>
      <td>3X03NfrYVLMXS2kz13t5WU</td>
      <td>Lets Kill Janice</td>
      <td>212661</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3O7HL3mzobCIxzErIMijxJ</td>
      <td>Spread A Little Sunshine</td>
      <td>3Ll91eivgGYSJUVnL1WV4h</td>
      <td>Playground</td>
      <td>3K6DKtqcjx8Vs5XEaTgZv7</td>
      <td>The Truth</td>
      <td>213466</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5La66MFvn7AX8ZjXu2s8l8</td>
      <td>Kalte Wut / Wenn Ich Einmal Reich Bin</td>
      <td>6u6ISXvYPQkgzAQ170HUO0</td>
      <td>Lucky Streik</td>
      <td>04QGT0sr98LHIUjqPJ4RG4</td>
      <td>Floh de Cologne</td>
      <td>313066</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3vUTaQZGXUTqgYWNUZiNW8</td>
      <td>Love's Been Good To Me</td>
      <td>27MGNEcpVfyejgypNIkogw</td>
      <td>One by One</td>
      <td>62JorFOkIjXHHU7GMT9r77</td>
      <td>Rod McKuen</td>
      <td>183493</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
query = 'DROP TABLE IF EXISTS BigTable;'
c.execute(query)
conn.commit()

c.execute(""" CREATE TABLE BigTable AS
            SELECT tracks.track,tracks.track_name, tracks.album, tracks.album_name, tracks.artist,
            tracks.artist_name, tracks.duration_ms, tracks.playlist_member, tracks.num_member,
            track_features.acousticness,track_features.danceability, track_features.energy,
            track_features.instrumentalness,track_features.key, track_features.liveness,track_features.loudness,
            track_features.mode, track_features.speechiness, track_features.tempo, track_features.time_signature,
            track_features.valence
            FROM tracks
            INNER JOIN track_features
            ON tracks.track = track_features.id""")

```





    <sqlite3.Cursor at 0x10f25f960>





```python
tracks_merged_df = psql.read_sql_query("SELECT * from BigTable ", conn)
tracks_merged_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>track_name</th>
      <th>album</th>
      <th>album_name</th>
      <th>artist</th>
      <th>artist_name</th>
      <th>duration_ms</th>
      <th>playlist_member</th>
      <th>num_member</th>
      <th>acousticness</th>
      <th>...</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1YGa5zwwbzA9lFGPB3HcLt</td>
      <td>Suavemente - Spanglish Edit</td>
      <td>6hGJXv2n5dNuIpaxpgSBOe</td>
      <td>Suavemente, The Remixes</td>
      <td>1c22GXH30ijlOfXhfLz9Df</td>
      <td>Elvis Crespo</td>
      <td>267333</td>
      <td>92453,112168,121648,238345,245267,303554,31985...</td>
      <td>12</td>
      <td>0.05310</td>
      <td>...</td>
      <td>0.791</td>
      <td>0.000485</td>
      <td>8</td>
      <td>0.2010</td>
      <td>-7.771</td>
      <td>1</td>
      <td>0.1160</td>
      <td>129.244</td>
      <td>4</td>
      <td>0.858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4gOMf7ak5Ycx9BghTCSTBL</td>
      <td>Sexo X Money (feat. Ozuna)</td>
      <td>3BB6mEEVz8vb2yMRSPHiDR</td>
      <td>8 Semanas</td>
      <td>5gUZ67Vzi1FV2SnrWl1GlE</td>
      <td>Benny Benni</td>
      <td>278809</td>
      <td>96862,117960</td>
      <td>2</td>
      <td>0.00336</td>
      <td>...</td>
      <td>0.779</td>
      <td>0.000001</td>
      <td>4</td>
      <td>0.3250</td>
      <td>-7.528</td>
      <td>0</td>
      <td>0.1900</td>
      <td>174.046</td>
      <td>4</td>
      <td>0.852</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7kl337nuuTTVcXJiQqBgwJ</td>
      <td>Jessica - Unedited Version</td>
      <td>1n9rbMLUmrXBBBJffS7pDj</td>
      <td>Brothers And Sisters</td>
      <td>4wQ3PyMz3WwJGI5uEqHUVR</td>
      <td>The Allman Brothers Band</td>
      <td>451160</td>
      <td>1231,2588,3054,4223,4777,4955,5094,5199,5991,6...</td>
      <td>871</td>
      <td>0.44700</td>
      <td>...</td>
      <td>0.855</td>
      <td>0.854000</td>
      <td>2</td>
      <td>0.1730</td>
      <td>-7.907</td>
      <td>1</td>
      <td>0.0340</td>
      <td>104.983</td>
      <td>4</td>
      <td>0.855</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0LAfANg75hYiV1IAEP3vY6</td>
      <td>Take Your Mama</td>
      <td>65Fllu4vQdZQOh6id0YwIM</td>
      <td>Scissor Sisters</td>
      <td>3Y10boYzeuFCJ4Qgp53w6o</td>
      <td>Scissor Sisters</td>
      <td>271906</td>
      <td>1697,2647,3150,5142,5199,6597,6669,7301,7367,7...</td>
      <td>534</td>
      <td>0.22000</td>
      <td>...</td>
      <td>0.954</td>
      <td>0.000018</td>
      <td>8</td>
      <td>0.0612</td>
      <td>-4.542</td>
      <td>1</td>
      <td>0.1210</td>
      <td>153.960</td>
      <td>4</td>
      <td>0.933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0Hpl422q9VhpQu1RBKlnF1</td>
      <td>3rd Chakra "Sacred Fire"</td>
      <td>07fAo1agwL9nuky8VmMhNa</td>
      <td>Chakradance</td>
      <td>3kbO7dxGm5uux3yrXYlnvR</td>
      <td>Jonthan Goldman</td>
      <td>509306</td>
      <td>231516</td>
      <td>1</td>
      <td>0.42200</td>
      <td>...</td>
      <td>0.834</td>
      <td>0.726000</td>
      <td>11</td>
      <td>0.1720</td>
      <td>-12.959</td>
      <td>1</td>
      <td>0.0631</td>
      <td>130.008</td>
      <td>4</td>
      <td>0.518</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>





```python
pd.read_sql_query("SELECT * from BigTable ", conn)
```




```python
# performing hierachical clustering
# train_tracks, test_tracks = train_test_split(tracks_merged_df, test_size=0.2, random_state=42)
train_tracks = tracks_merged_df
```




```python
def scale_datasets(train_data, cols_to_scale):
    train = train_data.copy()
#     test = test_data.copy()

    # fit the scaler on the training data
    scaler = StandardScaler().fit(train[cols_to_scale])

    # scale both the test and training data.
    train[cols_to_scale] = scaler.transform(train[cols_to_scale])
#     test[cols_to_scale] = scaler.transform(test[cols_to_scale])
    return train

# get columns that need to be scaled
not_to_scale = ['mode', 'track', 'track_name', 'album', 'album_name', 'artist', 'artist_name', 'playlist_member']
to_scale = train_tracks.columns.difference(not_to_scale)
# train_tracks_scaled, test_tracks_scaled = scale_datasets(train_tracks, test_tracks, to_scale)
train_tracks_scaled = scale_datasets(train_tracks, to_scale)
train_tracks_scaled.describe()
```


    /Users/andrafehmiu/Desktop/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py:617: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /Users/andrafehmiu/Desktop/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:9: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      if __name__ == '__main__':





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration_ms</th>
      <th>num_member</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.552714e-17</td>
      <td>-1.998401e-17</td>
      <td>-1.865175e-16</td>
      <td>-3.019807e-16</td>
      <td>-9.248158e-16</td>
      <td>3.108624e-17</td>
      <td>6.328271e-17</td>
      <td>5.995204e-17</td>
      <td>4.551914e-17</td>
      <td>0.690000</td>
      <td>-3.153033e-16</td>
      <td>-2.797762e-16</td>
      <td>5.717649e-17</td>
      <td>-1.815215e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>0.464823</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
      <td>1.005038e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.728999e+00</td>
      <td>-2.673091e-01</td>
      <td>-9.338852e-01</td>
      <td>-2.979480e+00</td>
      <td>-2.544541e+00</td>
      <td>-3.706251e-01</td>
      <td>-1.303501e+00</td>
      <td>-8.787226e-01</td>
      <td>-3.447685e+00</td>
      <td>0.000000</td>
      <td>-7.794134e-01</td>
      <td>-2.251437e+00</td>
      <td>-7.450531e+00</td>
      <td>-1.612685e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-4.968337e-01</td>
      <td>-2.599727e-01</td>
      <td>-8.631080e-01</td>
      <td>-5.617366e-01</td>
      <td>-6.800231e-01</td>
      <td>-3.706251e-01</td>
      <td>-1.031938e+00</td>
      <td>-6.307094e-01</td>
      <td>-5.163343e-01</td>
      <td>0.000000</td>
      <td>-6.611387e-01</td>
      <td>-7.300483e-01</td>
      <td>2.041241e-01</td>
      <td>-8.828781e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.783776e-01</td>
      <td>-2.350289e-01</td>
      <td>-3.411226e-01</td>
      <td>2.007501e-01</td>
      <td>3.261311e-01</td>
      <td>-3.705485e-01</td>
      <td>-2.172502e-01</td>
      <td>-4.567182e-01</td>
      <td>6.166226e-02</td>
      <td>1.000000</td>
      <td>-4.868391e-01</td>
      <td>3.691746e-02</td>
      <td>2.041241e-01</td>
      <td>-1.652379e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.350748e-01</td>
      <td>-1.396557e-01</td>
      <td>6.116665e-01</td>
      <td>7.185807e-01</td>
      <td>6.666392e-01</td>
      <td>-3.525086e-01</td>
      <td>8.690007e-01</td>
      <td>4.099448e-01</td>
      <td>7.721308e-01</td>
      <td>1.000000</td>
      <td>2.420501e-01</td>
      <td>6.311570e-01</td>
      <td>2.041241e-01</td>
      <td>8.271018e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.518389e+00</td>
      <td>8.075645e+00</td>
      <td>2.507656e+00</td>
      <td>1.899835e+00</td>
      <td>1.503722e+00</td>
      <td>3.295369e+00</td>
      <td>1.683689e+00</td>
      <td>3.924013e+00</td>
      <td>1.718260e+00</td>
      <td>1.000000</td>
      <td>3.442258e+00</td>
      <td>2.616494e+00</td>
      <td>2.755676e+00</td>
      <td>1.802332e+00</td>
    </tr>
  </tbody>
</table>
</div>





```python
def pca_x(data_train, to_scale, total_comp, var_thresh):
    data_train = data_train.set_index('track')
#     data_test = data_test.set_index('track')
    train_tracks_copy = data_train[to_scale].copy()
#     test_tracks_copy = data_test[to_scale].copy()

    # applying PCA
    pca = PCA(n_components= total_comp) # from our result in 1.5
    pca.fit(train_tracks_copy)

    # transforming train and test data
    x_train_pca = pca.transform(train_tracks_copy)
#     x_test_pca = pca.transform(test_tracks_copy)

    # plot pca var explained as function of number of PCs
    plt.plot(np.linspace(1, total_comp, total_comp), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('variance explained')
    plt.title('Cumulative variance explained by each component',fontsize=15)
    print("number of components that explain at least 90% of the variance=",\
    len(np.where(np.cumsum(pca.explained_variance_ratio_)<=var_thresh)[0])+1)

    return x_train_pca

x_train_pca = pca_x(train_tracks_scaled, to_scale, 13, 0.9)
```


    number of components that explain at least 90% of the variance= 10



![png](Connect_to_database_files/Connect_to_database_21_1.png)




```python
def find_optimal_cluster(pca_train):
    k_train = range(1, 40)
    knn_sum_squared_distances_train = []

    for k in k_train:
        knn_train = KMeans(n_clusters= k)
        knn_train = knn_train.fit(pca_train)
        knn_sum_squared_distances_train.append(knn_train.inertia_)

    plt.plot(k_train, knn_sum_squared_distances_train, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    return knn_sum_squared_distances_train

find_optimal_cluster = find_optimal_cluster(x_train_pca)
```



![png](Connect_to_database_files/Connect_to_database_22_0.png)




```python
find_optimal_cluster
```





    [1300.0000000000014,
     1112.3567077666132,
     1013.7593320850426,
     925.9152202928352,
     862.170942545108,
     784.5604459652873,
     742.7998712168622,
     684.6061949108164,
     643.9254867241616,
     618.2917525115306,
     598.3290903353958,
     567.9430750838708,
     547.5204716349906,
     513.3222979623332,
     499.76469985739027,
     473.42860822910376,
     450.6296176296054,
     436.7081828403112,
     418.54276993627815,
     394.66864312825214,
     374.2072411568014,
     363.70984794059484,
     342.1477195003828,
     333.7312830361003,
     311.3813603214546,
     311.5899239043024,
     305.62730211237863,
     283.078504169229,
     270.8778018776603,
     265.15120905179754,
     257.90679195230575,
     251.03538371256224,
     246.0103083314386,
     229.58199976249318,
     225.97844226851015,
     212.16918774449624,
     207.93712928003578,
     195.89591698569265,
     196.41892823492728]





```python
print(find_optimal_cluster[11:17])
```


    [567.9430750838708, 547.5204716349906, 513.3222979623332, 499.76469985739027, 473.42860822910376, 450.6296176296054]




```python
# based on the elbow plot, we pick k number of cluster to be 13
def optimal_knn_clustering(pca_train, k):
        knn_sum_squared_distances_train=[]
        knn_train = KMeans(n_clusters= k)
        knn_train = knn_train.fit(pca_train)
        # predict clusters
        labels = knn_train.predict(pca_train)
        # get cluster centers
        centers = knn_train.cluster_centers_
        return labels, centers

clusters, centers = optimal_knn_clustering(pca_train, 13)
clusters
```





    array([ 2,  2, 10,  2, 10,  0,  4,  4, 11,  2,  2,  9, 12, 12,  7, 11,  7,
            0,  8,  4, 11,  4,  2,  1, 11,  4,  1,  5, 12,  0,  2,  2, 11,  2,
            2,  0, 11,  6,  1, 11,  2, 11,  2,  2,  2, 11,  4, 11,  2,  1,  4,
            4,  2, 11, 10,  1,  4,  2,  2,  0,  1, 10, 11,  2,  0,  2,  2,  4,
            2,  7,  0,  0,  1,  2, 10,  0,  2,  1,  0,  1, 10,  2,  4,  1,  2,
            1, 11, 11,  2,  1,  2,  2,  3,  4,  1,  2,  4,  1,  4,  0],
          dtype=int32)





```python
import scipy.sparse as sparse
train_tracks['predicted cluster label'] = clusters.tolist()
train_tracks.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>track_name</th>
      <th>album</th>
      <th>album_name</th>
      <th>artist</th>
      <th>artist_name</th>
      <th>duration_ms</th>
      <th>playlist_member</th>
      <th>num_member</th>
      <th>acousticness</th>
      <th>...</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>predicted cluster label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1YGa5zwwbzA9lFGPB3HcLt</td>
      <td>Suavemente - Spanglish Edit</td>
      <td>6hGJXv2n5dNuIpaxpgSBOe</td>
      <td>Suavemente, The Remixes</td>
      <td>1c22GXH30ijlOfXhfLz9Df</td>
      <td>Elvis Crespo</td>
      <td>267333</td>
      <td>92453,112168,121648,238345,245267,303554,31985...</td>
      <td>12</td>
      <td>0.05310</td>
      <td>...</td>
      <td>0.000485</td>
      <td>8</td>
      <td>0.2010</td>
      <td>-7.771</td>
      <td>1</td>
      <td>0.1160</td>
      <td>129.244</td>
      <td>4</td>
      <td>0.858</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4gOMf7ak5Ycx9BghTCSTBL</td>
      <td>Sexo X Money (feat. Ozuna)</td>
      <td>3BB6mEEVz8vb2yMRSPHiDR</td>
      <td>8 Semanas</td>
      <td>5gUZ67Vzi1FV2SnrWl1GlE</td>
      <td>Benny Benni</td>
      <td>278809</td>
      <td>96862,117960</td>
      <td>2</td>
      <td>0.00336</td>
      <td>...</td>
      <td>0.000001</td>
      <td>4</td>
      <td>0.3250</td>
      <td>-7.528</td>
      <td>0</td>
      <td>0.1900</td>
      <td>174.046</td>
      <td>4</td>
      <td>0.852</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7kl337nuuTTVcXJiQqBgwJ</td>
      <td>Jessica - Unedited Version</td>
      <td>1n9rbMLUmrXBBBJffS7pDj</td>
      <td>Brothers And Sisters</td>
      <td>4wQ3PyMz3WwJGI5uEqHUVR</td>
      <td>The Allman Brothers Band</td>
      <td>451160</td>
      <td>1231,2588,3054,4223,4777,4955,5094,5199,5991,6...</td>
      <td>871</td>
      <td>0.44700</td>
      <td>...</td>
      <td>0.854000</td>
      <td>2</td>
      <td>0.1730</td>
      <td>-7.907</td>
      <td>1</td>
      <td>0.0340</td>
      <td>104.983</td>
      <td>4</td>
      <td>0.855</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0LAfANg75hYiV1IAEP3vY6</td>
      <td>Take Your Mama</td>
      <td>65Fllu4vQdZQOh6id0YwIM</td>
      <td>Scissor Sisters</td>
      <td>3Y10boYzeuFCJ4Qgp53w6o</td>
      <td>Scissor Sisters</td>
      <td>271906</td>
      <td>1697,2647,3150,5142,5199,6597,6669,7301,7367,7...</td>
      <td>534</td>
      <td>0.22000</td>
      <td>...</td>
      <td>0.000018</td>
      <td>8</td>
      <td>0.0612</td>
      <td>-4.542</td>
      <td>1</td>
      <td>0.1210</td>
      <td>153.960</td>
      <td>4</td>
      <td>0.933</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0Hpl422q9VhpQu1RBKlnF1</td>
      <td>3rd Chakra "Sacred Fire"</td>
      <td>07fAo1agwL9nuky8VmMhNa</td>
      <td>Chakradance</td>
      <td>3kbO7dxGm5uux3yrXYlnvR</td>
      <td>Jonthan Goldman</td>
      <td>509306</td>
      <td>231516</td>
      <td>1</td>
      <td>0.42200</td>
      <td>...</td>
      <td>0.726000</td>
      <td>11</td>
      <td>0.1720</td>
      <td>-12.959</td>
      <td>1</td>
      <td>0.0631</td>
      <td>130.008</td>
      <td>4</td>
      <td>0.518</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>





```python
# note! always close the connection at the end
conn.close()
```
