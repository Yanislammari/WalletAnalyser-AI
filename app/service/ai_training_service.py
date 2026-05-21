from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import hdbscan


from app.repositories.asset_cluster_repository import AssetClusterRepository
from app.service.data_visualisation import ExcelColAttributes, ModelColAttributes, clustering_overlay, correlation_show, pca_2D, print_cleaning_proc, run_pca, show_data, plot_umap_2d, visualize_pca
from app.service.get_finance_data import add_metrics_to_csv
from app.service.utils import winsorize

BASE_DIR = Path(__file__).resolve().parent

def drop_correlation(df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop([ExcelColAttributes.capex, ExcelColAttributes.ebitda, ExcelColAttributes.ebitda_trend, ExcelColAttributes.gross_margin, ExcelColAttributes.net_debt],axis=1) #drop correlation
    # correlation_show(df)
    return df

def compute_variable(df : pd.DataFrame) -> pd.DataFrame:
    ratio = df[ExcelColAttributes.net_debt] / df[ExcelColAttributes.ebitda]
    df[ModelColAttributes.net_debt_ebita] = np.where(
        df[ExcelColAttributes.ebitda] < 0,
        np.nan,
        ratio.clip(lower=0)
    )

    df[ModelColAttributes.total_asset_to_revenue] = df[ExcelColAttributes.total_asset] / df[ExcelColAttributes.revenue]
    df[ModelColAttributes.capex_to_revenue] = df[ExcelColAttributes.capex] / df[ExcelColAttributes.revenue]
    return df

def clean_data(file=BASE_DIR / "../data/metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(file)
    df = compute_variable(df)
    df = drop_correlation(df)

    if "uuid" not in df.columns:
        raise ValueError("Expected a 'uuid' column in the CSV")
    data_cols = [c for c in df.columns if c != ExcelColAttributes.uuid]

    mask = (
        df[data_cols].isna().all(axis=1)
        | df[ExcelColAttributes.year_pct_change].isna()
        | (df[ExcelColAttributes.operating_margin] == 0.0) & (df[ExcelColAttributes.ebitda_margin] == 0.0)
    )

    removed_count = mask.sum()

    df_clean = df.loc[~mask].copy()

    # 2. count rows that still have at least one NaN in non-uuid columns
    nan_rows_mask = df_clean.isna().any(axis=1)
    nan_rows_count = nan_rows_mask.sum()
    nan_per_column = df_clean[data_cols].isna().sum()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    # print_cleaning_proc(df_clean, removed_count, nan_rows_count, nan_per_column)

    return df_clean

def winsorize_data(df):
    df_wins = df.copy()
    df_wins = winsorize(df_wins, 
        [ExcelColAttributes.ebitda_level, ModelColAttributes.capex_to_revenue, ExcelColAttributes.growth_level,ExcelColAttributes.operating_margin]
        ,0.0035
    )
    df_wins = winsorize(df_wins, 
        [ExcelColAttributes.pe, ExcelColAttributes.price_to_book, ExcelColAttributes.peg, ModelColAttributes.total_asset_to_revenue]
        ,0.05)
    df_wins = winsorize(df_wins,[ModelColAttributes.net_debt_ebita, ExcelColAttributes.growth_trend, ExcelColAttributes.operating_margin],0.02)
    return df_wins

def impute_data(df):
    df = df.copy()
    model_columns = [c for c in df.select_dtypes(include="number").columns]

    for col in model_columns:
        global_median = df[col].median()

        def fill_value(group):
            # if group too small or all NaN → return NaN (handled later)
            if group[col].notna().sum() >= 5:
                return group[col].fillna(group[col].median())
            else:
                return group[col]

        # 1. sector-level fill
        df[col] = df.groupby("sector", group_keys=False).apply(fill_value)

        # 2. country-level fill (for remaining NaNs)
        def fill_country(group):
            if group[col].notna().sum() >= 5:
                return group[col].fillna(group[col].median())
            else:
                return group[col]

        df[col] = df.groupby("country", group_keys=False).apply(fill_country)

        # 3. global fallback
        df[col] = df[col].fillna(global_median)
    
    return df

def standardize_z_data(df_clean):
    df = df_clean.copy()
    model_columns = [c for c in df.select_dtypes(include="number").columns]
    df[model_columns] = (df[model_columns] - df[model_columns].mean()) / df[model_columns].std()

    return df

def run_kmeans(df, n_clusters=3):
    df = df.copy()
    model_columns = [c for c in df.select_dtypes(include="number").columns]
    X = df[model_columns]

    pca = PCA(n_components=0.85)  # keep 85% variance
    X_reduced = pca.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_reduced)
    df["cluster"] = labels

    print(silhouette_score(X_reduced, labels))

    pca = PCA()
    pca.fit(X)

    print(pca.explained_variance_ratio_)

    loadings = pd.DataFrame(
      pca.components_.T,
      columns=[f"PC{i+1}" for i in range(X.shape[1])],
      index=X.columns
    )

    print(loadings)

    return df, model

def run_gaussian(df_standardize, n_components=10):
    df = df_standardize.copy()
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    model_columns = [c for c in df.select_dtypes(include="number").columns]

    X = df[model_columns]
    labels = gmm.fit_predict(X)
    df["cluster"] = labels

    print(silhouette_score(X, labels))

    probs = gmm.predict_proba(X)

    for i, prob in enumerate(probs[:50]):
        formatted = ", ".join([f"{p:.3f}" for p in prob])
        print(f"Point {i}: [{formatted}]")

    return df, labels

def run_umap_hdbscan(df, 
                     umap_n_components=10, 
                     umap_n_neighbors=15,
                     umap_min_dist=0.1,
                     hdbscan_min_cluster_size=30,
                     hdbscan_min_samples=3):
    df = df.copy()
    model_columns = [c for c in df.select_dtypes(include="number").columns]
    X = df[model_columns].values

    # --- UMAP reduction ---
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=42,
        metric="euclidean"
    )
    X_reduced = reducer.fit_transform(X)

    # --- HDBSCAN clustering ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_method="leaf",
        prediction_data=True
    )
    labels = clusterer.fit_predict(X_reduced)

    if -1 in labels:
        soft = hdbscan.all_points_membership_vectors(clusterer)
        noise_mask = labels == -1
        labels[noise_mask] = soft[noise_mask].argmax(axis=1)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"Clusters found : {n_clusters}")
    print(f"Noise points   : {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    # silhouette only makes sense if we have ≥2 clusters and not too much noise
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        score = silhouette_score(X_reduced[mask], labels[mask])
        print(f"Silhouette score (excl. noise): {score:.4f}")
    else:
        print("Not enough clusters for silhouette score")

    df["cluster"] = labels

    plot_umap_2d(X, labels, umap_n_neighbors, umap_min_dist)
    return df, clusterer

if __name__ == "__main__":
  pd.set_option("display.max_rows", None)
  df_clean = clean_data()
  df_clean = df_clean.drop([ExcelColAttributes.year_pct_change], axis=1)
  print(df_clean.describe())

  df_winsorize = winsorize_data(df_clean)
  print(df_winsorize.describe())

  df_impute = impute_data(df_winsorize)
  print(df_impute.describe())

  df_standardize = standardize_z_data(df_impute)
  print(df_standardize.describe())
 
  pca_2D(df_standardize)
  visualize_pca(df_standardize)
  df_clustered, hdb_model = run_umap_hdbscan(
        df_standardize,
        umap_n_components=5,
        umap_n_neighbors=50,
        hdbscan_min_cluster_size=40,
        hdbscan_min_samples=1
    )
  clustering_overlay(df_clustered)
  show_data(df_clustered)
  run_pca(df_clustered)
