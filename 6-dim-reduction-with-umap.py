
import numpy as np
import pandas as pd
import umap
import matplotlib as pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde # for density-based plot

audio_UMAP_input = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/audio_UMAP_input.csv")
audio_UMAP_input_mini = audio_UMAP_input.iloc[0:40000, :] # running a tiny subset to save computational time

timestamps = audio_UMAP_input_mini.loc[:, 'timestamp']

audio_UMAP_input_mini = audio_UMAP_input_mini.to_numpy()

reducer = umap.UMAP(
    n_neighbors = 40, # bigger n_neighbors # means longer processing time
    n_components = 2,
    metric='euclidean',
    min_dist = 0.01,
    spread = 2, 
    n_jobs= -1
)

embedding = reducer.fit_transform(audio_UMAP_input_mini)

# print("Embedding DataType:")
# print(type(embedding))

print("starting UMAP plot...")
#plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
#plt.title('UMAP Projection: Hertz, Timestamp, Temp, Humidity, Centroid, Volume')
#plt.xlabel('UMAP1')
#plt.ylabel('UMAP2')
#plt.colorbar()
#plt.show()

x = embedding[:, 0]
y = embedding[:, 1]

# Calculate the density of points
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Plot with density-based coloring
# plt.scatter(x, y, c=z, cmap='viridis')  # 'viridis' is another color map that's good for showing gradients
plt.scatter(x, y, c=z, cmap='viridis')
plt.colorbar(label='Density')
plt.title('UMAP projection with Density-Based Coloring')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

# Plot with time-based coloring
# plt.scatter(x, y, c=timestamps, cmap='viridis')
# plt.colorbar(label='Progression of Time')
# plt.title('UMAP projection with Timestamp Coloring')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
plt.show()

embedding = pd.DataFrame(embedding)
embedding.to_csv("post_data_reduction_mini.csv")
print("saved csv")

