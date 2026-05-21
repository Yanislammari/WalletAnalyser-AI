import datetime

class AssetPrice:
    uuid: str
    asset_uuid: str
    asset_price: float
    asset_price_date: datetime
    created_at: datetime
    updated_at: datetime

    def __init__(self, dictionary):
      for k, v in dictionary.items():
        setattr(self, k, v)