from sqlalchemy import text

from app.core.db_connection import engine
from app.models.AssetPrice import AssetPrice

class AssetClusterRepository :

  table_name = 'AssetClusters'
  pass

  async def add_clusters_to_db(self, df):
      print(df.columns.tolist())  # what pandas has
      print(df.shape)             # how many rows
      with engine.connect() as con:
          con.execute(text(f'TRUNCATE TABLE "AssetClusters"'))
          con.commit() 

      df.to_sql(
          self.table_name,
          con=engine,
          if_exists="append",
          index=False
      )