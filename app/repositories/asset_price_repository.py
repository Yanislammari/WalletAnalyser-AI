from dataclasses import dataclass
import pandas as pd
from app.core.db_connection import engine
from app.models.AssetPrice import AssetPrice

@dataclass(frozen=True)
class AssetPriceAttributes:
  uuid : str = "uuid"
  asset_uuid: str = "asset_uuid"
  asset_price: str = "asset_price"
  asset_price_date: str = "asset_price_date"
  createdAt: str = "created_at"
  updatedAt: str = "updated_at"

class AssetPriceRepository :

  table_name = '"AssetPrices"'
  pass

  async def get_price_of_one_asset(self, asset_uuid) -> list[AssetPrice]:
    df = pd.read_sql(
        f'SELECT * FROM {self.table_name} WHERE {AssetPriceAttributes().asset_uuid} = %s',
        engine,
        params=(asset_uuid,)
    )
    records = df.to_dict(orient="records")
    return [AssetPrice(record) for record in records]