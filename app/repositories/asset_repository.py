from dataclasses import dataclass
import pandas as pd
from app.core.db_connection import engine
from app.models.Asset import Asset

@dataclass(frozen=True)
class AssetAttributes:
  uuid: str = "uuid"
  base_currency_uuid:str =  "base_currency_uuid"
  asset_type:str =  "asset_type"
  ticker_name:str =  "ticker_name"
  official_name:str =  "official_name"
  sector_uuid:str =  "sector_uuid"
  country_uuid:str =  "country_uuid"
  createdAt:str =  "created_at"
  updatedAt:str =  "updated_at"

class AssetRepository :

  table_name = '"Assets"'
  pass

  async def get_asset(self, asset_uuid) -> Asset:
    df = pd.read_sql(
        f'SELECT * FROM {self.table_name} WHERE {AssetAttributes().uuid} = %s',
        engine,
        params=(asset_uuid,)
    )
    records = df.to_dict(orient="records")
    if not records:
        return None
    return Asset(records[0])
