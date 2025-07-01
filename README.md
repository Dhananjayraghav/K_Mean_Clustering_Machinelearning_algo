# Customer_segmentation using k-means
K-means clustering is a popular unsupervised machine learning algorithm used to partition data into K distinct, non-overlapping clusters.

How It Works
Initialization: Randomly select K points as initial cluster centroids
Assignment: Assign each data point to the nearest centroid
Update: Recalculate centroids as the mean of all points in the cluster
Repeat: Iterate assignment and update steps until convergence (centroids stop changing)

ALGORITHM PSEUDOCODE:
1. Initialize k centroids randomly
2. While not converged:
   a. For each point:
      i. Calculate distance to all centroids
      ii. Assign to nearest centroid's cluster
   b. For each cluster:
      i. Recalculate centroid as mean of all points in cluster
   c. Check if centroids changed (convergence)
   
