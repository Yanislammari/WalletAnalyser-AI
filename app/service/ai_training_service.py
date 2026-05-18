from dataclasses import dataclass
from pathlib import Path
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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


BASE_DIR = Path(__file__).resolve().parent
    
def winsorize(df, cols, p=0.01):
    df = df.copy()
    for col in cols:
        lower = df[col].quantile(p)
        upper = df[col].quantile(1 - p)
        df[col] = np.clip(df[col], lower, upper)
    return df

def drop_correlation(df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop([ExcelColAttributes.capex, ExcelColAttributes.ebitda, ExcelColAttributes.ebitda_trend, ExcelColAttributes.gross_margin, # drop cause correlation
        ExcelColAttributes.net_debt, ExcelColAttributes.total_asset, ExcelColAttributes.revenue  # drop cause use elsewhere
                  ],axis=1)
    # corr = df.corr(numeric_only=True)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    # plt.show()
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
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # print(f"Rows removed (all non-uuid values NaN; or 52w dont exist;  ): {removed_count}")
    # print(f"Rows remaining: {len(df_clean)}")
    # print(f"Rows with at least one NaN: {nan_rows_count}")
    # print("\nNaN per column:")
    # print(nan_per_column.sort_values(ascending=False))

    return df_clean

def impute_data(df):
    df = df.copy()
    df = winsorize(df,
        [ExcelColAttributes.price_to_book, ExcelColAttributes.peg, ModelColAttributes.total_asset_to_revenue]
        ,0.06
    )
    df = winsorize(df, 
        [ExcelColAttributes.operating_margin, ExcelColAttributes.growth_trend, ExcelColAttributes.ebitda_level, ModelColAttributes.capex_to_revenue, ExcelColAttributes.growth_level]
        ,0.0035
    )
    df = winsorize(df, [ExcelColAttributes.pe],0.4)
    df = winsorize(df,[ModelColAttributes.net_debt_ebita],0.25)
    # print(df.describe())
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
    logs_cols = [ExcelColAttributes.peg, ExcelColAttributes.pe, ModelColAttributes.total_asset_to_revenue, ExcelColAttributes.operating_margin]
    exclude_cols = [ExcelColAttributes.year_pct_change, 
        ExcelColAttributes.ebitda_margin, ExcelColAttributes.growth_level, ExcelColAttributes.ebitda_level]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols and c not in logs_cols
    ]

    df[logs_cols] = np.log1p(df[logs_cols].abs())
    df[model_columns] = (df[model_columns] - df[model_columns].mean()) / df[model_columns].std()

    return df

def standardize_by_sector(df):
    df = df.copy()
    exclude_cols = [ExcelColAttributes.year_pct_change, ExcelColAttributes.ebitda_margin, ExcelColAttributes.operating_margin, ExcelColAttributes.growth_level, ExcelColAttributes.ebitda_level]
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

def run_gaussian(df):
    df = df.copy()
    gmm = GaussianMixture(n_components=5, covariance_type="full")
    exclude_cols = [ExcelColAttributes.year_pct_change]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols
    ]

    X = df[model_columns]
    labels = gmm.fit_predict(X)
    df["cluster"] = labels

    pca = PCA()
    pca.fit(X)

    print(pca.explained_variance_ratio_)

    loadings = pd.DataFrame(
      pca.components_.T,
      columns=[f"PC{i+1}" for i in range(X.shape[1])],
      index=X.columns
    )

    print(loadings)

    return df, labels
    
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


if __name__ == "__main__":
  pd.set_option("display.max_rows", None)
  df_clean = clean_data()
  # print(df_clean.describe())
  df_impute = impute_data(df_clean)
  print(df_impute.describe())
  df_standardize = standardize_z_data(df_impute)
  print(df_standardize.describe())
  df_clustered, kmeans_model = run_kmeans(df_standardize,7)
  run_elbow_method(df_clustered)
  print(df_clustered[["cluster"]].value_counts())
  print(df_clustered.groupby("cluster").mean(numeric_only=True))
  print(df_clustered[["uuid",ExcelColAttributes.name, "sector", "country", "cluster"]].head(50))
  run_pca(df_clustered)
