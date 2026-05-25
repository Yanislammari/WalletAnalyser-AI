import datetime

class Sector:
    uuid: str
    sector_name:str
    created_at: datetime
    updated_at: datetime

    def __init__(self, dictionary):
      for k, v in dictionary.items():
        setattr(self, k, v)