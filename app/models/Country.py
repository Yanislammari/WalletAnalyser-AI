import datetime

class Country:
    uuid: str
    country_name:str
    created_at: datetime
    updated_at: datetime

    def __init__(self, dictionary):
      for k, v in dictionary.items():
        setattr(self, k, v)