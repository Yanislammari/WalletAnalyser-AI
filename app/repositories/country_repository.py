from dataclasses import dataclass
import pandas as pd
from app.core.db_connection import engine
from app.models.Country import Country

@dataclass(frozen=True)
class CountryAttributes:
  uuid : str = "uuid"
  country_name: str = "country_name"
  createdAt: str = "created_at"
  updatedAt: str = "updated_at"

class CountryRepository :

  table_name = '"Countries"'
  pass

  async def get_country_uuid(self, country_name) -> Country:
    df = pd.read_sql(
        f'SELECT * FROM {self.table_name} WHERE {CountryAttributes().country_name} = %s',
        engine,
        params=(country_name,)
    )
    records = df.to_dict(orient="records")
    if not records:
        return None
    return Country(records[0])