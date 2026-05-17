from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

def clean_data(file=BASE_DIR / "../data/metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(file)

    if "uuid" not in df.columns:
        raise ValueError("Expected a 'uuid' column in the CSV")

    # columns to check (everything except uuid)
    data_cols = [c for c in df.columns if c != "uuid"]

    # 1. drop rows where ALL non-uuid columns are NaN
    empty_mask = df[data_cols].isna().all(axis=1)
    empty_mask = df['year_pct_change'].isna()
    removed_count = empty_mask.sum()

    df_clean = df.loc[~empty_mask].copy()

    # 2. count rows that still have at least one NaN in non-uuid columns
    nan_rows_mask = df_clean.isna().any(axis=1)
    nan_rows_count = nan_rows_mask.sum()
    nan_per_column = df_clean[data_cols].isna().sum()

    # print(f"Rows removed (all non-uuid values NaN; or 52w dont exist;  ): {removed_count}")
    # print(f"Rows remaining: {len(df_clean)}")
    # print(f"Rows with at least one NaN: {nan_rows_count}")
    # print("\nNaN per column:")
    # print(nan_per_column.sort_values(ascending=False))

    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    return df_clean

def impute_data(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
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

    # select only numeric columns (important!)
    numeric_cols = df.select_dtypes(include="number").columns

    # z-score standardization
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    return df

def standardize_by_sector(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns

    df[numeric_cols] = (
        df.groupby("sector")[numeric_cols]
          .transform(lambda x: (x - x.mean()) / x.std())
    )

    return df

def run_kmeans(df, n_clusters=3):
    df = df.copy()

    # keep only numeric features (no uuid, sector, country)
    X = df.select_dtypes(include="number")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X)

    return df, model

def run_pca(df_clustered):
  X = df_clustered.select_dtypes(include="number").drop(columns=["cluster"])
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)

  plt.scatter(components[:, 0], components[:, 1], c=df_clustered["cluster"])
  plt.title("K-means Clusters (PCA projection)")
  plt.show()

if __name__ == "__main__":
  pd.set_option("display.max_rows", None)
  df_clean = clean_data()
  df_impute = impute_data(df_clean)
  df_standardize = standardize_by_sector(df_impute)
  df_clustered, kmeans_model = run_kmeans(df_standardize, n_clusters=15)
  print(df_clustered[["cluster"]].value_counts())
  print(df_clustered.groupby("cluster").mean(numeric_only=True))
  print(df_clustered[["uuid", "sector", "country", "cluster"]].head(20))
  run_pca(df_clustered)
