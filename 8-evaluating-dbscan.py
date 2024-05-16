# April Ainsworth
# evaluates results of DBScan

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

post_data_reduction_mini = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/post_data_reduction_mini.csv")
post_data_reduction_mini = post_data_reduction_mini.to_numpy()

data = {
    'epsilon': [1.001, 1.001, 1.002, 1.002, 1.003, 1.003, 1.0032, 1.0032, 1.0034, 1.0034, 
                1.0035, 1.0038, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.01],
    'minPoints': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'clusters': [3762, 0, 2220, 0, 1299, 0, 1173, 0, 1055, 0, 1012, 898, 817, 552, 413, 309, 241, 189, 153],
    'noise': [10676, 40000, 4927, 40000, 2917, 40000, 2679, 40000, 2456, 40000, 2350, 2066, 1915, 1335, 915, 613, 446, 307, 223]
}

df = pd.DataFrame(data)

# Plot clusters and noise against epsilon
fig, ax1 = plt.subplots()

ax1.set_xlabel('epsilon')
ax1.set_ylabel('clusters', color='tab:blue')
ax1.plot(df['epsilon'], df['clusters'], color='tab:blue', label='clusters')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('noise', color='tab:red')
ax2.plot(df['epsilon'], df['noise'], color='tab:red', label='noise')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Clusters and Noise vs Epsilon')
plt.show()

# Assuming you have your dataset in variable `X` (features only, without labels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(post_data_reduction_mini)

# Calculate silhouette score for each parameter combination
for index, row in df.iterrows():
    dbscan = DBSCAN(eps=row['epsilon'], min_samples=int(row['minPoints'])) 
    labels = dbscan.fit_predict(X_scaled)
    if len(set(labels)) > 1:  # Silhouette score requires at least 2 clusters
        score = silhouette_score(X_scaled, labels)
        print(f"Epsilon: {row['epsilon']}, MinPoints: {row['minPoints']} -> Silhouette Score: {score}")
    else:
        print("Length <= 1")

