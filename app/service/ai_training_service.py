from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class ExcelColAttributes:
    uuid: str = "uuid"
    price_to_book: str = "price_to_book"
    peg: str = "peg"
    forward_pe: str = "forward_pe"
    country: str = "country"
    sector: str = "sector"
    ebitda_margin: str = "ebitda_margin"
    gross_margin: str = "gross_margin"
    year_pct_change: str = "year_pct_change"
    growth_level: str = "growth_level"
    growth_trend: str = "growth_trend"
    ebitda_level: str = "ebitda_level"
    ebitda_trend: str = "ebitda_trend"
    net_debt_ebitda: str = "net_debt_ebitda"
    capex_to_revenue: str = "capex_to_revenue"
    total_asset_to_revenue: str = "total_asset_to_revenue"
    name: str = "name"

BASE_DIR = Path(__file__).resolve().parent
    

def winsorize(df, cols, p=0.01):
    df = df.copy()
    for col in cols:
        lower = df[col].quantile(p)
        upper = df[col].quantile(1 - p)
        df[col] = np.clip(df[col], lower, upper)
    return df

def clean_data(file=BASE_DIR / "../data/metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(file)

    if "uuid" not in df.columns:
        raise ValueError("Expected a 'uuid' column in the CSV")

    # columns to check (everything except uuid)
    data_cols = [c for c in df.columns if c != "uuid"]

    # 1. drop rows where ALL non-uuid columns are NaN
    data_cols = [c for c in df.columns if c != ExcelColAttributes.uuid]

    mask = (
        df[data_cols].isna().all(axis=1)
        | df[ExcelColAttributes.year_pct_change].isna()
        | (df[ExcelColAttributes.gross_margin] == 0.0) & (df[ExcelColAttributes.ebitda_margin] == 0.0)
    )

    removed_count = mask.sum()

    df_clean = df.loc[~mask].copy()

    # 2. count rows that still have at least one NaN in non-uuid columns
    nan_rows_mask = df_clean.isna().any(axis=1)
    nan_rows_count = nan_rows_mask.sum()
    nan_per_column = df_clean[data_cols].isna().sum()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # print(f"Rows removed (all non-uuid values NaN; or 52w dont exist;  ): {removed_count}")
    # print(f"Rows remaining: {len(df_clean)}")
    # print(f"Rows with at least one NaN: {nan_rows_count}")
    # print("\nNaN per column:")
    # print(nan_per_column.sort_values(ascending=False))
    df_test = winsorize(df_clean, [ExcelColAttributes.price_to_book, ExcelColAttributes.peg, ExcelColAttributes.forward_pe, ExcelColAttributes.growth_trend, ExcelColAttributes.ebitda_trend])
    print(df_clean.describe())
    #print(df_test.describe())

    return df_clean

def impute_data(df):
    df = df.copy()
    exclude_cols = [ExcelColAttributes.year_pct_change]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols
    ]

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
    exclude_cols = [ExcelColAttributes.year_pct_change]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols
    ]

    # select only numeric columns (important!)
    iqr = df[model_columns].quantile(0.75) - df[model_columns].quantile(0.25)

    # z-score standardization
    df[model_columns] = (df[model_columns] - df[model_columns].median()) / iqr
    #df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    return df

def standardize_by_sector(df):
    df = df.copy()
    exclude_cols = [ExcelColAttributes.year_pct_change, ExcelColAttributes.ebitda_margin, ExcelColAttributes.gross_margin]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols
    ]

    df[model_columns] = (
        df.groupby("sector")[model_columns]
          .transform(lambda x: (x - x.mean()) / x.std())
    )

    return df

def run_kmeans(df, n_clusters=3):
    df = df.copy()
    exclude_cols = [ExcelColAttributes.year_pct_change]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols
    ]

    X = df[model_columns]

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X)

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
  df_standardize = standardize_z_data(df_impute)
  df_clustered, kmeans_model = run_kmeans(df_impute, n_clusters=6)
  # print(df_clustered[["cluster"]].value_counts())
  # print(df_clustered.groupby("cluster").mean(numeric_only=True))
  # print(df_clustered[["uuid",ExcelColAttributes.name, "sector", "country", "cluster"]].head(50))
  # run_pca(df_clustered)
