import os
os.environ["UMAP_FORCE_DENSE"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap

import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

df = pd.read_csv("ship_performance_cleaned.csv")

df_numeric = df.select_dtypes(include=["int64", "float64"])

print("Shape before scaling:", df_numeric.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.scatter(X_umap[:, 0], X_umap[:, 1], s=10, c='blue')
plt.title("UMAP Projection - Ship Performance")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

umap_df = pd.DataFrame(X_umap, columns=["UMAP_1", "UMAP_2"])
umap_df.to_csv("ship_umap_output.csv", index=False)

print("\n UMAP reduced data saved as 'ship_umap_output.csv'")
print("Shape after reduction:", umap_df.shape)
