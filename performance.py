import pandas as pd
import joblib
from sklearn.metrics import silhouette_score

df = pd.read_csv('ship_umap_output.csv')
X_umap = df.values  

dbscan = joblib.load("kmeans_umap_model.pkl")

labels = dbscan.labels_

if len(set(labels)) > 1 and -1 not in set(labels):
    score = silhouette_score(X_umap, labels)
    print(f"Silhouette Score: {score:.4f}")
else:
    print("Silhouette Score cannot be computed — only one cluster or mostly noise (-1).")