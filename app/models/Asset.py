import datetime

class Asset:
    uuid: str
    base_currency_uuid:str
    asset_type:str
    ticker_name:str
    official_name:str 
    sector_uuid:str
    country_uuid:str
    created_at: datetime
    updated_at: datetime

    def __init__(self, dictionary):
      for k, v in dictionary.items():
        setattr(self, k, v)