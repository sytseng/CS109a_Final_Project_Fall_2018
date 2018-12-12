---
title: EDA
notebook: EDA_alkmini.ipynb
nav_include: 0
---


```python
import sys, os
import sqlite3
import pandas as pd
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```


## Exploration of the Spotify API data - audio features

In order to understand the audio features, we can look at the distribution of the data provided by the Spotify API.



```python
# create a new database by connecting to it
conn = sqlite3.connect("spotifyDB.db")
cur = conn.cursor()
```




```python
# read the table to ensure that everything is working
pd.read_sql_query("select * from tracks WHERE num_member > 10 LIMIT 5;", conn)
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
      <td>4HPBYMNnpP9Yc8u51VUyaD</td>
      <td>Mariconatico</td>
      <td>5bZ7Jgk7YE35ULHWO3wgHP</td>
      <td>Sincopa</td>
      <td>07PdYoE4jVRF6Ut40GgVSP</td>
      <td>Cartel De Santa</td>
      <td>269333</td>
      <td>1065,9003,10561,15769,21028,138066,147268,1567...</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3MCMHJ0p0tD0eRQC0Wg7Wu</td>
      <td>Headshot!</td>
      <td>3k4aYVzOplQp6MTye2KlQ0</td>
      <td>Unbreakable</td>
      <td>4Uc6biA3i00oRAWvIfjhlk</td>
      <td>MyChildren MyBride</td>
      <td>210813</td>
      <td>3647,14323,35549,71836,74701,98691,113560,1189...</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2F950w5JasEsPPHfspO5yn</td>
      <td>Know Your Onion - Spotify Sessions Curated by ...</td>
      <td>5aeSTpjqwuarRBM1zPCbOY</td>
      <td>Spotify Sessions</td>
      <td>4LG4Bs1Gadht7TCrMytQUO</td>
      <td>The Shins</td>
      <td>152653</td>
      <td>2136,5176,27542,55583,77394,85407,118872,18360...</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5GAU7SP15OkPwj2tjXlUJn</td>
      <td>Goodfriend</td>
      <td>2kyzu2gZ1SlkrzW6l4b7dD</td>
      <td>Feathers &amp; Fishhooks</td>
      <td>251UrhgNbMr15NLzQ2KyKq</td>
      <td>Rayland Baxter</td>
      <td>266746</td>
      <td>3626,8788,17107,49060,52900,53173,64212,66550,...</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7bTNGfXSJthLunBJzgOyBw</td>
      <td>Harder</td>
      <td>5nL2ZC8NBSjMZfgpOp9P0w</td>
      <td>FLYTRAP</td>
      <td>41yEdWozNYEzA2RfgYQHgr</td>
      <td>CJ Fly</td>
      <td>177842</td>
      <td>17432,30221,51929,53295,74242,86976,109832,139...</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



We can see from the table above that get the unique track id, album id, artist id, as well as the convential names of these attributes, and information about the song length and playlist. We think it will be useful to connect to the Spotify API to get more information about the songs on a song by song basis. We do this below.



```python
# Access spotify
import spotipy as sp
import spotipy.oauth2 as oauth2
import sqlite3
import pandas as pd

# set up authorization token
credentials = oauth2.SpotifyClientCredentials(
        client_id='153369a05314402294db1a574caaff2a',
        client_secret='c6fff0923a0c44c5851fc4415038e8fa')

token = credentials.get_access_token()
spotify = sp.Spotify(auth=token)
```




```python
# Explore 500 tracks

N = 2000

audio_feat_tracks = pd.read_sql_query("SELECT * from tracks ORDER BY RANDOM() LIMIT {};".format(N), conn)

```




```python
# Take track IDs in order to access the API

ids = audio_feat_tracks['track'].values.tolist()

# All audio features -- for EDA

danceability = []
energy = []
key = []
loudness = []
mode = []
speechiness = []
acousticness = []
instrumentalness = []
liveness = []
valence = []
tempo = []
duration_ms = []
time_signature = []

features = [danceability, energy, key, loudness, mode, speechiness, acousticness,\
           instrumentalness, liveness, valence, tempo, duration_ms, time_signature]

feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',\
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', \
                 'duration_ms', 'time_signature']

# Collect into lists:

for track_id in ids:
    results = spotify.audio_features(tracks=track_id)

    # Add into lists
    danceability.append(results[0]['danceability'])
    energy.append(results[0]['energy'])
    key.append(results[0]['key'])
    loudness.append(results[0]['loudness'])
    mode.append(results[0]['mode'])
    speechiness.append(results[0]['speechiness'])
    acousticness.append(results[0]['acousticness'])
    instrumentalness.append(results[0]['instrumentalness'])
    liveness.append(results[0]['liveness'])
    valence.append(results[0]['valence'])
    tempo.append(results[0]['tempo'])
    duration_ms.append(results[0]['duration_ms'])
    time_signature.append(results[0]['time_signature'])


```


    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs
    retrying ...1secs




```python
# Plot distributions

N_features = len(feature_names)

fig, ax = plt.subplots(ncols = 2, nrows = N_features//2+1, figsize = (14,25))
for i in range(N_features):
    ax[i//2, i%2].hist(features[i]);
    ax[i//2, i%2].set_title(feature_names[i])
    ax[i//2, i%2].set_ylabel('Frequency')
    ax[i//2, i%2].set_xlabel('Values')

plt.tight_layout()

```



![png](EDA_alkmini_files/EDA_alkmini_8_0.png)


We notice that the distributions for danceability and tempo are somewhat normal, whereas the valence distribution is almost uniform. For the other variables, some of which are categorical, and others are continuous with varying distributions. This information will be useful to keep in mind when building our models.


---- ----- ----- -----
