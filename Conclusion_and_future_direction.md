---
title: Conclusion and Future Directions
notebook: Conclusion_and_future_direction.ipynb
nav_include: 8
---

# Conclusion and future direction

In this project, we worked on the Million Playlists Dataset and audio features of tracks from Spotify API to implement song recommender systems. We approached this goal using to different ways: ***content-based*** and ***collaborative filerting***.

## Content-based filtering
From our analysis using only the Spotify API and the audio features of the songs, we saw that it would be difficult to make a model purely based on the song content.

This conclusion, nonetheless, is not unreasonable.
Energy, for instance, which describes the intensity of the song, can have similar values for songs that are under very different genres, e.g. metal and reggaeton.
Therefore, metrics like these are not strongly correlated with the similarity of two songs, or the probability of them belonging in the same playlist.

As a result, our analysis based on content illuminated the necessity of explorative analysis on collaborative filtering.

## K-means Clustering
We used k-nn clustering to identify which tracks were similar based on their audio features. Although our model provided a content-based filtering approach, it was not very robust because the clustering of tracks was not stable. Because of the way the algorithm is built, if clustering is applied repeatedly to different samples from this distribution, it might sometimes construct the horizontal and sometimes the vertical solution. Obviously, these two solutions are very different from each other, hence the clustering results are instable. Therefore, the challenge with applying the k-nn clustering algorithm to recommend songs, we would have to ensure the correct number of clusters is chosen -- otherwise, the model becomes unstable.

## Collaborative filtering
***
We used both ***matrix factorization*** and ***neural networks*** to implement collaborative filtering. We tested them on a subset of 10,000 playlists seeded with a popular song as well as a randomly selected subset, and used Jaccard index as a performance measurement.

### Matrix factorization


We used singular value decomposition (SVD) for matrix factorization, testing with various number of latent factors. We found that for the seeded dataset, a small number of latent factors (5-50) was sufficient to make song recommendation that seemed to be "similar" (measured by the Jaccard index) to existing songs in a given playlist. When we applied SVD on the random subset, we needed a larger number of latent factors. The recommended songs were less similar to the existing songs for a given playlist, but the Jaccard index was still higher than random choices.

### Neural collaborative filtering

We built a neural network (NNCF) to perform collaborative filtering. The network embedded high dimensional playlist/track data into a low dimensional space, and performed MF and MLP mixing in separate streams before combining the information and making final prediction. The NNCF trained on the seeded subset achieved 82% sensitivity and 94% specificity, and the Jaccard index of recommeded songs were similar to the existing songs. For the random subset, the network performance wasn't as good as the seeded ubset, but it still made song recommendation that had higher Jaccard index than random choice. Further expansion of the network structure, more training data or extended training time might be needed to increase network performance.

### Comparison and future direction

Both matrix factorization and neural collaborative filtering seemed to give comparable results on the data we tested. In terms of computational cost, SVD is lower and faster to perform. However, neural networks have the flexibility to incorporate new data and extra information. We can expand the playlist or track number but changing the input-to-embedding structure on a network with pre-trained weights, or engineer the network to take temporal information (user history) as well, such as recent hits of specific songs. Therefore, there is a great potnetial to improve the neural network approach for song recommendation.

A better version of recommender system should be a hybrid between both content-based and collaborative filtering. This could be done in a hierarchical way: using content-based methods to define clusters of tracks which limits the serach range, and then adopting collaborative filtering to generate recommendations. Alternatively, the neural collaborative filtering could also incorporate features of tracks and input, and incorporate those information into a coherent recommender system.
