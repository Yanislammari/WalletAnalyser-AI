from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.metrics import euclidean_distances
from sklearn.mixture import GaussianMixture
import umap

@dataclass(frozen=True)
class ExcelColAttributes:
    uuid: str = "uuid"
    price_to_book: str = "price_to_book"
    peg: str = "peg"
    pe: str = "pe"
    country: str = "country"
    sector: str = "sector"
    ebitda_margin: str = "ebitda_margin"
    gross_margin: str = "gross_margin"
    operating_margin: str = "operating_margin"
    year_pct_change: str = "year_pct_change"
    growth_level: str = "growth_level"
    growth_trend: str = "growth_trend"
    ebitda_level: str = "ebitda_level"
    ebitda_trend: str = "ebitda_trend"
    ebitda: str = "ebitda"
    net_debt: str = "net_debt"
    revenue: str = "revenue"
    capex: str = "capex"
    total_asset: str = "total_asset"
    name: str = "name"

@dataclass(frozen=True)
class ModelColAttributes:
    net_debt_ebita:str = "net_debt_ebitda" 
    capex_to_revenue:str = "capex_to_revenue"
    total_asset_to_revenue:str = "total_asset_to_revnue"


def print_cleaning_proc(df_clean, removed_count, nan_rows_count, nan_per_column):

  print(f"Rows removed (all non-uuid values NaN; or 52w dont exist;  ): {removed_count}")
  print(f"Rows remaining: {len(df_clean)}")
  print(f"Rows with at least one NaN: {nan_rows_count}")
  print("\nNaN per column:")
  print(nan_per_column.sort_values(ascending=False))


def run_pca(df_clustered):
  X = df_clustered.select_dtypes(include="number").drop(columns=["cluster"])
  pca = PCA(n_components=0.9)
  components = pca.fit_transform(X)

  plt.scatter(components[:, 0], components[:, 1], c=df_clustered["cluster"])
  plt.title("K-means Clusters (PCA projection)")
  plt.show()

def run_elbow_method(df_test):
    df_clustered = df_test.copy()
    X = df_clustered.select_dtypes(include="number")
    inertias = []
    K = range(2, 11)

    for k in K:
        model = KMeans(k, n_init=10, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.plot(K, inertias, marker="o")
    plt.title("distance")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.show()
    
def show_data(df_clustered):
  print(df_clustered[["cluster"]].value_counts())
  print(df_clustered.groupby("cluster").mean(numeric_only=True))
  print(df_clustered[["uuid",ExcelColAttributes.name, "sector", "country", ExcelColAttributes.ebitda_margin, ExcelColAttributes.operating_margin,"cluster"]].head(50))
  print(df_clustered[df_clustered["cluster"] == 0])

def print_cluster_distance_matrix(dist_matrix):
    df = pd.DataFrame(dist_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis")

    plt.title("Inter-Cluster Distance Matrix")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")

    plt.show()

def cluster_analysis(kmeans):
  centers = kmeans.cluster_centers_
  k = len(centers)

  dist_matrix = np.zeros((k, k))

  for i in range(k):
      for j in range(k):
          dist_matrix[i, j] = euclidean(centers[i], centers[j])

  np.set_printoptions(precision=3, suppress=True)
  print(dist_matrix)
  print_cluster_distance_matrix(dist_matrix)

def correlation_show(df):
  corr = df.corr(numeric_only=True)
  plt.figure(figsize=(10, 8))
  sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
  plt.show()

def inspect_uuid(df_clustered, kmeans):

    target_uuids = [
       "25d07549-3610-46f5-bd19-2ec3dd216b4c", # pepsi
       "ec29c132-7ec0-4f0b-b219-246d983c8d94", # alphabet
       "06e3dc19-b941-41a6-9fb6-a456c8fed84a", # alibaba
       "16c73542-2f79-409d-bbdf-22d74db22515", # p&g
    ]

    for target_uuid in target_uuids:
      print("\n" + "=" * 80)
      row_df = df_clustered[df_clustered[ExcelColAttributes.uuid] == target_uuid]
      if row_df.empty:
          print(f"{target_uuid} not found")
          continue

      row_df = row_df.iloc[0]
      cluster_id = row_df["cluster"]
      print(f"{row_df[ExcelColAttributes.name]}")
      print(f"{row_df}")
      model_columns = [c for c in df_clustered.select_dtypes(include="number").columns]

      vector = df_clustered.loc[
          df_clustered[ExcelColAttributes.uuid] == target_uuid,
          model_columns
      ].values

      distances = euclidean_distances(
          vector,
          kmeans.cluster_centers_
      )[0]

      # print("\nNearest clusters:")

      # nearest = np.argsort(distances)

      # for cid in nearest[:10]:
      #     marker = " <-- assigned" if cid == cluster_id else ""
      #     print(f"Cluster {cid}: {distances[cid]:.4f}{marker}")

      centroid = kmeans.cluster_centers_[cluster_id]
      diff = vector.flatten() - centroid

      importance = pd.Series(diff, index=model_columns)

      print("Top drivers (absolute impact):")
      print(
          importance
          .sort_values(key=abs, ascending=False)
          .head(50)
      )

def pca_2D(df_test):
  df = df_test.copy()
  pca = PCA(n_components=5)
  model_columns = [c for c in df.select_dtypes(include="number").columns]
  X = df[model_columns]
  X_pca = pca.fit_transform(X)
  plt.figure(figsize=(8,6))
  plt.scatter(X_pca[:,0], X_pca[:,1])
  plt.title("PC1 vs PC2")
  plt.show()

  plt.scatter(X_pca[:,0], X_pca[:,2])
  plt.title("PC1 vs PC3")
  plt.show()

  plt.scatter(X_pca[:,1], X_pca[:,2])
  plt.title("PC2 vs PC3")
  plt.show()

def clustering_overlay(df_test):
  df = df_test.copy()
  pca = PCA(n_components=5)
  labels = df["cluster"]
  model_columns = [c for c in df.select_dtypes(include="number").columns]
  X = df[model_columns]
  X_pca = pca.fit_transform(X)
  plt.figure(figsize=(8,6))
  plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", alpha=0.7)
  plt.title("PC1 vs PC2")
  plt.show()

  plt.scatter(X_pca[:,0], X_pca[:,2], c=labels, cmap="viridis", alpha=0.7)
  plt.title("PC1 vs PC3")
  plt.show()

  plt.scatter(X_pca[:,1], X_pca[:,2], c=labels, cmap="viridis", alpha=0.7)
  plt.title("PC2 vs PC3")
  plt.show()

def plot_umap_2d(X, labels, umap_n_neighbors=15, umap_min_dist=0.1):
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=42
    )
    X_2d = reducer_2d.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("UMAP 2D — HDBSCAN clusters")
    plt.tight_layout()
    plt.show()

def visualize_pca(df):
    model_columns = [c for c in df.select_dtypes(include="number").columns]
    X = df[model_columns]
    pca = PCA()
    pca.fit(X)

    print(pca.explained_variance_ratio_)

    loadings = pd.DataFrame(
      pca.components_.T,
      columns=[f"PC{i+1}" for i in range(X.shape[1])],
      index=X.columns
    )

    print(loadings)
