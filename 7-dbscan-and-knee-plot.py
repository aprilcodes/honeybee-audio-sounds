# April Ainsworth
# runs DBSCAN and plots k-distance

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

post_data_reduction_mini = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/post_data_reduction_mini.csv")
#post_data_reduction_mini = post_data_reduction_mini.to_numpy()

# k-distance plot 

# neighbors = NearestNeighbors(n_neighbors=2)
# neighbors_fitted = neighbors.fit(post_data_reduction_mini)
# distances, indices = neighbors_fitted.kneighbors(post_data_reduction_mini) # indices won't be used
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.xlabel('Indices of the Data Points')
# plt.ylabel('Distance to Nearest Neighbors')
# plt.title('Nearest Neighbors Plot To Determine Epsilon')
# plt.show()
 
# general rule: choose epsilon based on the "knee" of the curve
epsilon = 1.01

# general rule: set minPts = to dimensionality of data plus 1 or 2 (may be different for audio data)

minPoints = 3

dbscan = DBSCAN(eps=epsilon, min_samples=minPoints)

clusters = dbscan.fit_predict(post_data_reduction_mini) # a cluster label of -1 represents noise

print("epsilon:", epsilon)
print("minPoints:", minPoints)

print("Number of clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
print("Number of noise points:", list(clusters).count(-1))