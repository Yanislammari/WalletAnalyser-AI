import asyncio
from datetime import datetime
from pathlib import Path
import uuid

import pandas as pd
import umap.umap_ as umap
import hdbscan

from app.repositories.asset_cluster_repository import AssetClusterRepository
from app.repositories.asset_repository import AssetRepository
from app.repositories.country_repository import CountryRepository
from app.repositories.sector_repository import SectorRepository
from app.service.ai_training_service import clean_data, impute_data, standardize_z_data, winsorize_data
from app.service.data_visualisation import ExcelColAttributes, ModelColAttributes
from app.service.get_finance_data import add_metrics_to_csv

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "data/metrics.csv"

async def complete_db():
    df = pd.read_csv(path)
    
    tasks = [process_row(row) for _, row in df.iterrows()]
    await asyncio.gather(*tasks)

async def process_row(row):
    uuid = row["uuid"]
    asset = await AssetRepository().get_asset(uuid)

    if asset.sector_uuid is None and pd.notna(row[ExcelColAttributes.sector]):
        sector = await SectorRepository().get_sector_uuid(row[ExcelColAttributes.sector])
        if sector is not None:
            await AssetRepository().patch_sector(uuid, sector.uuid)

    if asset.country_uuid is None and pd.notna(row[ExcelColAttributes.country]):
        country = await CountryRepository().get_country_uuid(row[ExcelColAttributes.country])
        if country is not None:
            await AssetRepository().patch_country(uuid, country.uuid)

async def save_df_to_db(df_clustered):
    # format for db
    df_clustered = df_clustered.drop([ ExcelColAttributes.sector, ExcelColAttributes.country, ExcelColAttributes.name,
        ExcelColAttributes.ebitda, ExcelColAttributes.net_debt, ExcelColAttributes.total_asset, ExcelColAttributes.capex
    ], axis = 1)
    df_clustered = df_clustered.rename(columns={
        "uuid": "asset_uuid",
    })
    data_cols = [c for c in df_clustered.columns if c != "asset_uuid"]
    mask = (
        df_clustered[data_cols].isna().all(axis=1) | 
        df_clustered[ExcelColAttributes.year_pct_change].isna() |
        (df_clustered[ExcelColAttributes.operating_margin] == 0.0) & (df_clustered[ExcelColAttributes.ebitda_margin] == 0.0)
        )
    df_clean = df_clustered[~mask]
    df_clean[ExcelColAttributes.uuid] = [str(uuid.uuid4()) for _ in range(len(df_clean))] 
    df_clean["created_at"] = datetime.now()
    df_clean["updated_at"] = datetime.now()
    await AssetClusterRepository().add_clusters_to_db(df=df_clean)
    return

def prod_create_hdbscan(df, 
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

    df_csv = pd.read_csv(BASE_DIR / "data/metrics.csv")
    df_csv["cluster"] = pd.NA
    df_csv[ModelColAttributes.capex_to_revenue] = df[ModelColAttributes.capex_to_revenue]
    df_csv[ModelColAttributes.net_debt_ebita] = df[ModelColAttributes.net_debt_ebita]
    df_csv[ModelColAttributes.total_asset_to_revenue] = df[ModelColAttributes.total_asset_to_revenue]
    df_csv.loc[df.index, "cluster"] = labels
    add_metrics_to_csv(df_csv, BASE_DIR / "data/metrics_cluster.csv")
    return df_csv, clusterer

async def create_prod_model():
    print("Completing db with sector and country uuids...")
    await complete_db()
    print("Training model ...")
    df_clean = clean_data(file=BASE_DIR / "data/metrics.csv")
    df_clean = df_clean.drop([ExcelColAttributes.year_pct_change], axis=1)
    df_winsorize = winsorize_data(df_clean)
    df_impute = impute_data(df_winsorize)
    df_standardize = standardize_z_data(df_impute)
    df_clustered, hdb_model = prod_create_hdbscan(
        df_standardize,
        umap_n_components=5,
        umap_n_neighbors=50,
        hdbscan_min_cluster_size=40,
        hdbscan_min_samples=1
    )
    print("Model ready !!")
    await save_df_to_db(df_clustered)
    print("Model saved to db")
    return df_clustered

if __name__ == "__main__":
    asyncio.run(create_prod_model())