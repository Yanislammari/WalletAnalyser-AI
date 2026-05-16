import asyncio
import os
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
import yfinance as yf
from app.repositories.asset_price_repository import AssetPriceRepository
from app.repositories.asset_repository import AssetRepository

msft_uuid = '79094066-a318-4727-9937-bfe30108132a'

BASE_DIR = Path(__file__).resolve().parent
file = BASE_DIR / "../data/metrics.csv"

def add_metrics_to_csv(metrics : dict):
  if os.path.exists(file):
    df = pd.read_csv(file)
  else:
    df = pd.DataFrame()

  new_row = pd.DataFrame([metrics])
  if "name" in df.columns:
    df = df[df["name"] != metrics["name"]]

  df = pd.concat([df, new_row], ignore_index=True)
  df.to_csv(file, index=False)

def extract_ttm_info(info: dict) -> dict:
    return {
       "price_to_book" :  info["priceToBook"],
       "peg" : info["pegRatio"],
       "forward_pe" : info["forwardPE"],
       "country" : info["country"],
       "sector" : info["sector"],
       "ebitda_margin" : info["ebitdaMargins"],
       "gross_margin" : info["grossMargins"],
       "year_pct_change" : info["52WeekChange"]
    }

def growth_calculation(rev_y : dict, rev_q : dict) -> dict :
    # helper: safe check for quarterly validity
    quarterly_valid = (
        rev_q is not None
        and len(rev_q) == 5
        and not rev_q.isna().any()
    )

    yearly_valid = (
        len(rev_y) == 4
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
    rev_y = financials.loc["Total Revenue"].iloc[:4]
    rev_q = quarterly_financials.loc["Total Revenue"].iloc[:5]

    growth_total = growth_calculation(rev_y, rev_q)
    print(growth_total)
    growth_level = np.median(growth_total[0:-2])
    growth_trend = growth_total.iloc[0] - growth_total.iloc[-2]

    #EBITDA
    ebitda_y = financials.loc["EBITDA"].iloc[:4]
    ebitda_q = quarterly_financials.loc["EBITDA"].iloc[:5]

    ebitda_total = growth_calculation(ebitda_y, ebitda_q)
    
    ebitda_level = np.median(ebitda_total[0:-2])
    ebitda_trend = ebitda_total.iloc[0] - ebitda_total.iloc[-2]

    return {
        "growth_level": growth_level,
        "growth_trend" : growth_trend,
        "ebitda_level": ebitda_level,
        "ebitda_trend": ebitda_trend,
    }

def compute_balance_sheet_metrics(t : yf.Ticker, ) -> dict :
  balance_sheet = t.quarterly_balance_sheet
  info = t.info

  revenue_ttm = info["totalRevenue"]
  net_debt = balance_sheet.loc["Net Debt"].iloc[:4]
  net_debt_ebitda = (
    np.nan
    if net_debt.isna().any()
    else net_debt.sum() / info["ebitda"]
  )

  total_asset = balance_sheet.loc["Total Assets"].iloc[:4]
  total_asset_to_revenue = (
      np.nan
      if total_asset.isna().any()
      else ( balance_sheet.loc["Total Assets"].iloc[:4].sum() / 4 ) / revenue_ttm
  )

  capex = t.quarterly_cash_flow.loc["Capital Expenditure"].iloc[:4]
  capex_to_revenue = (
    np.nan
    if capex.isna().any()
    else capex.sum() / revenue_ttm * -1
  )

  return {
    "net_debt_ebitda": net_debt_ebitda,
    "capex_to_revenue": capex_to_revenue,
    "total_asset_to_revenue": total_asset_to_revenue
  }


async def main():
  asset = await AssetRepository().get_asset(msft_uuid)
  t = yf.Ticker(asset.ticker_name)

  info_ttm = extract_ttm_info(t.info)
  balance_sheet_metrics = compute_balance_sheet_metrics(t) 
  growth_metrics = compute_growth_metrics(t)
  pd.set_option('display.max_rows', None)
  features = {
    "name" : t.info["displayName"],
    **info_ttm,
    **growth_metrics,
    **balance_sheet_metrics,
  }
  clean_features = {
    k: float(v) if isinstance(v, (np.floating, float, int)) else v
    for k, v in features.items()
  }
  add_metrics_to_csv(clean_features)
  pprint.pprint(clean_features)



if __name__ == "__main__":
    asyncio.run(main())