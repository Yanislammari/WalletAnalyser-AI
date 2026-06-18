import asyncio
import os
from pathlib import Path
import pprint
from typing import Any
import numpy as np
import pandas as pd
import yfinance as yf
from app.repositories.asset_price_repository import AssetPriceRepository
from app.repositories.asset_repository import AssetRepository

BASE_DIR = Path(__file__).resolve().parent

def add_metrics_to_csv(metrics: list, file = BASE_DIR / "../data/metrics.csv"):
    open(file, "w").close()
    new_df = pd.DataFrame(metrics)
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        new_df.to_csv(file, index=False)
        return

    # every N inserts:
    df = pd.read_csv(file)
    df = pd.concat([df, new_df], ignore_index=True)
    df["uuid"] = df["uuid"].astype(str).str.strip()
    df = df.drop_duplicates("uuid", keep="last")
    df.to_csv(file, index=False)

def get_key_value(info : dict, key : str) -> Any:
  data = (
    info[key]
    if info is not None and key in info
    else np.nan
  )
  return data

def get_key(info : pd.DataFrame, key : str, endIndex = 4 ) -> pd.Series | pd.DataFrame:
  data = (
    info.loc[key].iloc[:endIndex]
    if info is not None and key in info.index
    else np.nan
  )
  return data

def extract_ttm_info(info: dict) -> dict:
    net_debt = get_key_value(info,'totalDebt') - get_key_value(info, 'totalCash')
    return {
       "price_to_book" :  get_key_value(info,"priceToBook"),
       "peg" : get_key_value(info,"pegRatio"),
       "pe" : get_key_value(info,"trailingPE"),
       "country" : get_key_value(info,"country"),
       "sector" : get_key_value(info,"sector"),
       "ebitda_margin" : get_key_value(info,"ebitdaMargins"),
       "gross_margin" : get_key_value(info,"grossMargins"),
       "operating_margin" : get_key_value(info,"operatingMargins"),
       "year_pct_change" : get_key_value(info,"52WeekChange"),

       "net_debt" : net_debt,
       "revenue" : get_key_value(info,"totalRevenue"),
       "ebitda" : get_key_value(info,"ebitda"),
    }

def growth_calculation(rev_y : dict, rev_q : dict) -> dict :
    # helper: safe check for quarterly validity
    quarterly_valid = (
        isinstance(rev_q, pd.Series)
        and len(rev_q) == 5
        and not rev_q.isna().any()
        and not rev_q.iloc[4] == 0
    )

    yearly_valid = (
        isinstance(rev_y, pd.Series)
        and len(rev_y) == 4
        and not rev_y.isna().any()
    )

    if not yearly_valid:
        growth_total = np.nan

    elif not quarterly_valid or str(rev_q.index[4].date()) == str(rev_y.index[0].date()):
        growth_total = rev_y.pct_change(-1)

    else:
        growth_q = rev_q.iloc[0] / rev_q.iloc[4] - 1
        growth_y = rev_y.pct_change(-1)

        growth_total = pd.concat([
            growth_y,
            pd.Series({rev_q.index[0]: growth_q})
        ]).sort_index(ascending=False)

    return growth_total

def compute_growth_metrics(t : yf.Ticker) -> dict : 
    financials = t.financials
    quarterly_financials = t.quarterly_financials

    # Total Revenue
    rev_y = get_key(financials,"Total Revenue")
    rev_q = get_key(quarterly_financials,"Total Revenue",5)

    growth_total = growth_calculation(rev_y, rev_q)
    if growth_total is np.nan:
       growth_level = np.nan
       growth_trend = np.nan
    else :
      growth_level = np.median(growth_total[0:-2])
      growth_trend = (growth_total.iloc[0] - growth_total.iloc[-2]) / growth_total.iloc[-2]

    #EBITDA
    operating_y = get_key(financials,"Operating Revenue")
    operating_q = get_key(quarterly_financials,"Operating Revenue",5)
    depre_y = get_key(financials, "Reconciled Depreciation")
    depre_q = get_key(financials, "Reconciled Depreciation")
    interest_expense_y = get_key(financials, "Interest Expense")
    interest_expense_q = get_key(financials, "Interest Expense")

    ebitda_y = (
      operating_y
      + depre_y
      - interest_expense_y
    )
    ebitda_q = (
      operating_q
      + depre_q
      - interest_expense_q
    )

    ebitda_total = growth_calculation(ebitda_y, ebitda_q)
    
    if ebitda_total is np.nan:
      ebitda_level = np.nan
      ebitda_trend = np.nan
    else:
      ebitda_level = np.median(ebitda_total[0:-2])
      ebitda_trend = (ebitda_total.iloc[0] - ebitda_total.iloc[-2] ) / ebitda_total.iloc[-2]

    return {
        "growth_level": growth_level,
        "growth_trend" : growth_trend,
        "ebitda_level": ebitda_level,
        "ebitda_trend": ebitda_trend,
    }

def compute_balance_sheet_metrics(t : yf.Ticker, ) -> dict :
  quarterly_balance_sheet = t.quarterly_balance_sheet

  total_asset = get_key(quarterly_balance_sheet, "Total Assets")
  total_asset = (
      np.nan
      if total_asset is np.nan
      else ( total_asset.sum() / 4 )
  )

  capex = get_key(t.quarterly_cash_flow, "Capital Expenditure")
  capex = (
    np.nan
    if capex is np.nan
    else capex.sum() * -1
  )

  return {
    "capex": capex,
    "total_asset": total_asset
  }


async def extract_stocks_metrics(uuid : str):
  asset = await AssetRepository().get_asset(uuid)
  print(asset.ticker_name , asset.uuid)
  t = yf.Ticker(asset.ticker_name)

  info_ttm = extract_ttm_info(t.info)
  balance_sheet_metrics = compute_balance_sheet_metrics(t) 
  growth_metrics = compute_growth_metrics(t)
  features = {
    "uuid" : asset.uuid,
    "name" : asset.official_name,
    **info_ttm,
    **growth_metrics,
    **balance_sheet_metrics,
  }
  clean_features = {
    k: float(v) if isinstance(v, (np.floating, float, int)) else v
    for k, v in features.items()
  }
  return clean_features

async def fetch_data_for_ai():
    pd.set_option("display.max_rows", None)
    assetRepository = AssetRepository()
    assets = await assetRepository.get_all_uuid()
    res = []
    for asset in assets:
      features = await extract_stocks_metrics(asset["uuid"])
      res.append(features)
      await asyncio.sleep(1)

    add_metrics_to_csv(res)

if __name__ == "__main__":
    asyncio.run(fetch_data_for_ai())