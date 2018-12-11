
## Subsampling playlists and tracks 
***
This is the script we used to subsample playlist-track contingency from database, both randomly selecting, or seeding with a popular song.
***
Import libraries.



```python
import sqlite3
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy
import pickle
%matplotlib inline
```


Connect to database.



```python
# create a new database by connecting to it
conn = sqlite3.connect("spotifyDB.db")
cur = conn.cursor()
```


### Random sampling
***
Here we randomly selected 10,000 playlists and found all tracks in the playlists.



```python
# subsample playlists
n_lists = 10000
query = 'SELECT playlist_id FROM playlists ORDER BY RANDOM() LIMIT {};'.format(n_lists)
sublist = pd.read_sql_query(query, conn)
sublist = list(sublist['playlist_id'].values)

tracks = []
for playlist_id in sublist:
    query = 'SELECT tracks FROM playlists WHERE playlist_id = {};'.format(playlist_id)
    all_tracks = pd.read_sql_query(query, conn)
    ids = all_tracks['tracks'].values[0].split(',')
    tracks = list(set(tracks) | set(ids))

n_tracks = len(tracks)
```


### Sampling with a seed track
***
Here we randomly selected a popular track that is member of >10,000 playlists, selected 10,000 playlists from it, and found all tracks in all those playlists.



```python
# subsample playlists from a popular seed track
n_lists = 10000
seed = pd.read_sql_query("select track, playlist_member from tracks WHERE num_member >10000 ORDER BY RANDOM() LIMIT 1;", conn)

seed_track = seed['track'].values[0]

sublist = seed['playlist_member'].values[0].split(',')
sublist = sublist[:n_lists]

tracks = []
for playlist_id in sublist:
    query = 'SELECT tracks FROM playlists WHERE playlist_id = {};'.format(playlist_id)
    all_tracks = pd.read_sql_query(query, conn)
    ids = all_tracks['tracks'].values[0].split(',')
    tracks = list(set(tracks) | set(ids))

n_tracks = len(tracks)
sublist = [int(ii) for ii in sublist]
seed_ind = tracks.index(seed_track)
```


### Find all the playlist-track pairs
***
We idenitified all playlist-track pairs with their indices.



```python
# fetch playlist_member for all tracks
keys = tracks
keys = '\',\''.join(keys)
keys = "('"+keys+"')"

query = 'SELECT playlist_member FROM tracks WHERE track IN {};'.format(keys)
all_members = pd.read_sql_query(query, conn)

for i,track_id in enumerate(tracks):
    mem = all_members['playlist_member'].values[i].split(',')
    mem = [int(track) for track in mem]
    # find indices of playlists that this track is a member of
    list_ind = [idx for idx, list_id in enumerate(sublist) if list_id in mem]
    # create coordinate and values for the sparse matrix
    n_col = len(list_ind)
    if i == 0:
        col = np.array(list_ind).reshape(1,-1)
        row = i*np.ones(n_col,).astype('int').reshape(1,-1)
        value = np.ones(n_col,).reshape(1,-1)
    elif i > 0:
        col = np.hstack((col, np.array(list_ind).reshape(1,-1)))
        row = np.hstack((row, i*np.ones(n_col,).astype('int').reshape(1,-1)))
        value = np.hstack((value, np.ones(n_col,).reshape(1,-1)))
```


### Create a sparse matrix 
***
We filled all the playlist-track pairs into a sparse matrix and save them.



```python
# create the sparse matrix
sps_acc = sps.coo_matrix((value.reshape(-1,), (row.reshape(-1,), col.reshape(-1,))), shape=(n_tracks, n_lists))
```




```python
# for seeded data, save the seed index
# seed_ind = tracks.index(seed_track)
```




```python
# save files
scipy.sparse.save_npz('sparse_10000_rand.npz', sps_acc)
with open('sublist_10000_rand.pkl', 'wb') as f1:
    pickle.dump(sublist, f1)
with open('tracks_10000_rand.pkl', 'wb') as f2:
    pickle.dump(tracks, f2)    
# with open('seed_ind.pkl', 'wb') as f3:
#     pickle.dump(seed_ind, f3) 
```


### Close cursor and disconnect to the database



```python
# disconnect
cur.close()
conn.close()
```

