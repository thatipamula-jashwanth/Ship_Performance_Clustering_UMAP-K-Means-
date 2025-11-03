import pandas as pd
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv('ship_umap_output.csv')
X_umap = df.values

kmeans = KMeans(
    n_clusters=8,       
    init='k-means++',   
    random_state=42
)
kmeans.fit(X_umap)

joblib.dump(kmeans, "kmeans_umap_model.pkl")
print('Model trained and saved as kmeans_umap_model.pkl')
