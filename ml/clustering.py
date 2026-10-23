import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import os

base_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

output_dir = os.path.join(base_dir, "output")

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(
    os.path.join(
        base_dir,
        "output",
        "cleaned_library_circulation.csv"
    )
)

df["circulation_count"] = pd.to_numeric(
    df["circulation_format_total"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "location_acronym",
        "resource_type",
        "circulation_count"
    ]
)

data = df.pivot_table(
    index="location_acronym",
    columns="resource_type",
    values="circulation_count",
    aggfunc="sum",
    fill_value=0
)

print("Location Resource Matrix")
print(data)

scaler = StandardScaler()

scaled = scaler.fit_transform(
    data
)

best_k = 2
best_score = -1

max_k = min(
    6,
    len(data) - 1
)

for k in range(2, max_k + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        scaled
    )

    score = silhouette_score(
        scaled,
        labels
    )

    print(
        "K =",
        k,
        "Silhouette Score =",
        round(score, 4)
    )

    if score > best_score:
        best_score = score
        best_k = k

print(
    "\nBest number of clusters:",
    best_k
)

model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(
    scaled
)

result = data.copy()

result["cluster"] = labels

result.to_csv(
    os.path.join(output_dir, "location_clusters.csv")
)

print("\nCluster Results:")
print(result)

print(
    "\nSaved:"
)

print(
    os.path.join(output_dir, "location_clusters.csv")
)