from dataclasses import dataclass
import pandas as pd
from app.core.db_connection import engine
from app.models.Sector import Sector

@dataclass(frozen=True)
class SectorAttributes:
  uuid : str = "uuid"
  sector_name: str = "sector_name"
  createdAt: str = "created_at"
  updatedAt: str = "updated_at"

class SectorRepository :

  table_name = '"Sectors"'
  pass

  async def get_sector_uuid(self, sector_name) -> Sector:
    df = pd.read_sql(
        f'SELECT * FROM {self.table_name} WHERE {SectorAttributes().sector_name} = %s',
        engine,
        params=(sector_name,)
    )
    records = df.to_dict(orient="records")
    if not records:
        return None
    return Sector(records[0])