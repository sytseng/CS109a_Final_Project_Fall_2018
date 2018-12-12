---
title: A Real Example
notebook: Real_example_song_recommendation_with_NNCF.ipynb
nav_include: 7
---

## Song recommendation in action
***
Here we demonstrated an example of song recommendation using NNCF model.


### Load libraries



```python
import sqlite3
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy
import pickle
import keras
from keras.models import load_model
```


    Using TensorFlow backend.


### Connect to database



```python
# connect to database
conn = sqlite3.connect("spotifyDB.db")
```


### Load data and model



```python
# load files
sps_acc = scipy.sparse.load_npz('sparse_10000.npz')
with open('sublist.pkl', 'rb') as f1:
    sublist = pickle.load(f1)
with open('tracks.pkl', 'rb') as f2:
    tracks = pickle.load(f2)

# convert matrix to csc type
sps_acc = sps_acc.tocsc()
sps_acc
```





    <74450x10000 sparse matrix of type '<class 'numpy.float64'>'
    	with 1030015 stored elements in Compressed Sparse Column format>





```python
# load the best NNCF model
model_file = "models_10000/model.220-0.30-0.88.h5"
model = load_model(model_file)

# find track and playlist number
n_tracks, n_playlists = sps_acc.shape[0], sps_acc.shape[1]
```




## Make song recommendations to a popular playlist
***
Here we identified the most popular playlist in the subset by searching the playlist with maximal number of followers. It was a playlist called No Limit with 212 tracks in it and had 232 followers.



```python
keys = [str(pl) for pl in sublist]
keys = '\',\''.join(keys)
keys = "('"+keys+"')"

# fetch information of first 10 popular playlists
query = 'SELECT playlist_id,playlist_name,num_tracks, num_followers \
FROM playlists WHERE playlist_id IN {} ORDER BY num_followers DESC LIMIT 10;'.format(keys)
playlist_df = pd.read_sql_query(query, conn)
display(playlist_df)
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
      <th>playlist_id</th>
      <th>playlist_name</th>
      <th>num_tracks</th>
      <th>num_followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>276338</td>
      <td>No Limit</td>
      <td>212</td>
      <td>232</td>
    </tr>
    <tr>
      <th>1</th>
      <td>126788</td>
      <td>No Heart</td>
      <td>116</td>
      <td>227</td>
    </tr>
    <tr>
      <th>2</th>
      <td>148574</td>
      <td>Needed Me</td>
      <td>230</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12870</td>
      <td>Desiigner — Panda</td>
      <td>184</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306214</td>
      <td>Shabba</td>
      <td>209</td>
      <td>76</td>
    </tr>
    <tr>
      <th>5</th>
      <td>220058</td>
      <td>Old playlist.</td>
      <td>242</td>
      <td>74</td>
    </tr>
    <tr>
      <th>6</th>
      <td>274306</td>
      <td>No Limit</td>
      <td>25</td>
      <td>68</td>
    </tr>
    <tr>
      <th>7</th>
      <td>187065</td>
      <td>🔥lit🔥</td>
      <td>167</td>
      <td>58</td>
    </tr>
    <tr>
      <th>8</th>
      <td>90643</td>
      <td>bonfire playlist</td>
      <td>243</td>
      <td>32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>271904</td>
      <td>Hip Hop</td>
      <td>51</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>


Now we make recommendation to this playlist using our NNCF model.



```python
# find the corresponding position of the playlist in our subset
this_id = sublist.index(playlist_df.playlist_id.values[0])

# make recommendation using NNCF model
rec, _, real = recommend_tracks(sps_acc, this_id, model)
```


Here are the first 20 songs in this playlist.



```python
# obtain track uri for the first 20 tracks in the playlist
member = [tr for idx, tr in enumerate(tracks) if idx in list(real[:20])]
keys = '\',\''.join(member)
keys = "('"+keys+"')"

# fetch information of existing tracks
query = 'SELECT track_name, album_name, artist_name FROM tracks WHERE track IN {};'.format(keys)
display(pd.read_sql_query(query, conn))
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
      <th>track_name</th>
      <th>album_name</th>
      <th>artist_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Frontin' - Club Mix</td>
      <td>The Neptunes Present... Clones</td>
      <td>Pharrell Williams</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R.O.C.K. In The U.S.A. (A Salute To 60's Rock)</td>
      <td>Scarecrow</td>
      <td>John Mellencamp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No Te Quiero En Mi Vida</td>
      <td>1392</td>
      <td>David Lee Garza</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No Money (Remix)</td>
      <td>No Money (Remix)</td>
      <td>Vapi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7 Oaks</td>
      <td>The Turnpike Troubadours</td>
      <td>Turnpike Troubadours</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sly Fox - Original Mix</td>
      <td>The Adventures of Mr. Fox</td>
      <td>Koan Sound</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bonbon - English Version</td>
      <td>Bonbon</td>
      <td>Era Istrefi</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fancy - Yellow Claw Remix</td>
      <td>Fancy</td>
      <td>Iggy Azalea</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Thugged Out - feat. Foxx [Explicit Album Version]</td>
      <td>Incarcerated</td>
      <td>Boosie Badazz</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Winner Take All</td>
      <td>The Pursuit of Nappyness</td>
      <td>Nappy Roots</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Slippin</td>
      <td>Slippin</td>
      <td>Waka Flocka Flame</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DKC 2 - Forest Interlude</td>
      <td>Donkey Kong Country Vol.3</td>
      <td>Good Knight Productions</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Old School Hyphy (feat. Nef the Pharaoh)</td>
      <td>Old School Hyphy (feat. Nef the Pharaoh)</td>
      <td>Corn</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kiss It</td>
      <td>Kiss It</td>
      <td>DEV</td>
    </tr>
    <tr>
      <th>14</th>
      <td>We Fall Again</td>
      <td>Pink Season</td>
      <td>Pink Guy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>I'm Too Sexy</td>
      <td>Fredhead</td>
      <td>Right Said Fred</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Pain</td>
      <td>Futures</td>
      <td>Jimmy Eat World</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Don't Trust Me</td>
      <td>Don't Trust Me</td>
      <td>2ONE!2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Quiet Little Voices</td>
      <td>These Four Walls</td>
      <td>We Were Promised Jetpacks</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Phantom of the Opera (From "Phantom of the Ope...</td>
      <td>On and Off Stage</td>
      <td>Lesley Garrett</td>
    </tr>
  </tbody>
</table>
</div>


Here are the top 10 songs that the model recommended!



```python
# obtain track uir for the top 10 recommended songs
rec_tracks = [tr for idx, tr in enumerate(tracks) if idx in list(rec[:10])]
keys = '\',\''.join(rec_tracks)
keys = "('"+keys+"')"

# fetch information of recommended songs
query = 'SELECT track_name, album_name, artist_name FROM tracks WHERE track IN {};'.format(keys)
display(pd.read_sql_query(query, conn))
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
      <th>track_name</th>
      <th>album_name</th>
      <th>artist_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Call Me</td>
      <td>Call Me</td>
      <td>NEIKED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Almeria</td>
      <td>Almeria</td>
      <td>Everydayz</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Where Is The Love?</td>
      <td>Elephunk</td>
      <td>The Black Eyed Peas</td>
    </tr>
    <tr>
      <th>3</th>
      <td>On verra</td>
      <td>Feu</td>
      <td>Nekfeu</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jolene</td>
      <td>The Foundation</td>
      <td>Zac Brown Band</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Knotty Head</td>
      <td>Imperial</td>
      <td>Denzel Curry</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Strangeland</td>
      <td>All 6's And 7's</td>
      <td>Tech N9ne</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Seeing Two</td>
      <td>Soda Lime Love</td>
      <td>MIAMIGO</td>
    </tr>
    <tr>
      <th>8</th>
      <td>You Don't Know Me (feat. Hemi)</td>
      <td>Slow Motion</td>
      <td>Jarren Benton</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Maca.Frama.Lamma</td>
      <td>Da U.S. Open</td>
      <td>Mac Dre And Mac Mall</td>
    </tr>
  </tbody>
</table>
</div>





