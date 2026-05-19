from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes

from app.service.data_visualisation import ExcelColAttributes, ModelColAttributes, cluster_analysis, correlation_show, print_cleaning_proc, run_elbow_method, run_pca, show_data
from app.service.utils import classify_net_debt_ebitda, winsorize, winsorize_two_sides

BASE_DIR = Path(__file__).resolve().parent

def drop_correlation(df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop([ExcelColAttributes.capex, ExcelColAttributes.ebitda, ExcelColAttributes.ebitda_trend, ExcelColAttributes.gross_margin, # drop cause correlation
        ExcelColAttributes.net_debt, ExcelColAttributes.total_asset, ExcelColAttributes.revenue # drop cause use elsewhere
                  ],axis=1)
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
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
    print_cleaning_proc(df_clean, removed_count, nan_rows_count, nan_per_column)

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

def boost_data(df):
    df = df.copy()
    # df[ExcelColAttributes.operating_margin] = df[ExcelColAttributes.operating_margin] * 1.2
    # df[ExcelColAttributes.ebitda_margin] = df[ExcelColAttributes.ebitda_margin] * 1.2
    # df[ExcelColAttributes.growth_level] = df[ExcelColAttributes.growth_level] * 2
    # df[ExcelColAttributes.growth_trend] = df[ExcelColAttributes.growth_trend] * 2

    return df


def standardize_z_data(df_clean):
    df = df_clean.copy()
    logs_cols = [] # [ExcelColAttributes.ebitda_level] #[ExcelColAttributes.peg, ModelColAttributes.total_asset_to_revenue, ExcelColAttributes.operating_margin, ModelColAttributes.net_debt_ebita]
    exclude_cols = [ExcelColAttributes.year_pct_change
                    #ExcelColAttributes.growth_trend, ExcelColAttributes.growth_level, ExcelColAttributes.ebitda_trend, ExcelColAttributes.ebitda_level
    ]
    model_columns = [
      c for c in df.select_dtypes(include="number").columns
      if c not in exclude_cols and c not in logs_cols
    ]

    df[logs_cols] = np.log1p(df[logs_cols].abs())
    df[model_columns] = (df[model_columns] - df[model_columns].mean()) / df[model_columns].std()

    return df

def standardize_by_sector(df):
    df = df.copy()
    exclude_cols = [ExcelColAttributes.year_pct_change] #, ExcelColAttributes.ebitda_margin, ExcelColAttributes.operating_margin, ExcelColAttributes.growth_level, ExcelColAttributes.ebitda_level]
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
    
def run_kproto(df, n_clusters=3):
    exclude_cols = [ExcelColAttributes.uuid, ExcelColAttributes.country, ExcelColAttributes.sector, ExcelColAttributes.name, ExcelColAttributes.year_pct_change]
    clustering_cols = [
        c for c in df.columns
        if c not in exclude_cols
    ]
    clustering_df = df[clustering_cols].copy()

    categorical_cols = [ModelColAttributes.net_debt_ebita]

    categorical_columns = [
        clustering_df.columns.get_loc(col)
        for col in categorical_cols
    ]
    X = clustering_df
    clustering_df = clustering_df.to_numpy()

    kproto = KPrototypes(n_clusters, random_state=42)
    clusters = kproto.fit_predict(
        clustering_df,
        categorical=categorical_columns
    )

    df["cluster"] = clusters

    pca = PCA()
    pca.fit(X)

    print(pca.explained_variance_ratio_)

    loadings = pd.DataFrame(
      pca.components_.T,
      columns=[f"PC{i+1}" for i in range(X.shape[1])],
      index=X.columns
    )

    print(loadings)

    return df, clusters


if __name__ == "__main__":
  pd.set_option("display.max_rows", None)
  df_clean = clean_data()
  print(df_clean.describe())

  df_winsorize = winsorize_data(df_clean)
  print(df_winsorize.describe())

  df_impute = impute_data(df_winsorize)
  print(df_impute.describe())

  df_standardize = standardize_z_data(df_impute)
  print(df_standardize.describe())

  df_boost = boost_data(df_standardize)
  print(df_boost.describe())

  df_clustered, kmeans_model = run_kmeans(df_boost,10)
  run_elbow_method(df_clustered)
  show_data(df_clustered)
  run_pca(df_clustered)
  cluster_analysis(kmeans_model)
