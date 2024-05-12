
import numpy as np
import pandas as pd
import umap
import matplotlib as pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde # for density-based plot

audio_UMAP_input_standardized = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/audio_UMAP_input_standardized.csv")

reducer = umap.UMAP(
    n_neighbors=5,
    n_components=2,
    metric='euclidean',
    min_dist=0.12,
    spread=9.999 # was 10, got divide by zero warning
    # random_state=42
)

embedding = reducer.fit_transform(audio_UMAP_input_standardized)

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
plt.scatter(x, y, c=z, cmap='viridis')  # 'viridis' is another color map that's good for showing gradients
plt.colorbar(label='Density')
plt.title('UMAP projection with Density-based Coloring')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()
