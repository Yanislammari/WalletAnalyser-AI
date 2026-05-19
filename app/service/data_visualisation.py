from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

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
  X = df_clustered.select_dtypes(include="number").drop(columns=["cluster",ExcelColAttributes.year_pct_change])
  pca = PCA(n_components=0.9)
  components = pca.fit_transform(X)

  plt.scatter(components[:, 0], components[:, 1], c=df_clustered["cluster"])
  plt.title("K-means Clusters (PCA projection)")
  plt.show()

def run_elbow_method(df_clustered):
    X = df_clustered.select_dtypes(include="number").drop(columns=["cluster",ExcelColAttributes.year_pct_change])
    inertias = []
    K = range(2, 11)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.plot(K, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.show()
    
def show_data(df_clustered):
  print(df_clustered[["cluster"]].value_counts())
  print(df_clustered.groupby("cluster").mean(numeric_only=True))
  print(df_clustered[["uuid",ExcelColAttributes.name, "sector", "country", ExcelColAttributes.growth_level, ExcelColAttributes.growth_trend,"cluster"]].head(50))

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