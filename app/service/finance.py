import asyncio
import pprint
import pandas as pd
import yfinance as yf
from app.repositories.asset_price_repository import AssetPriceRepository
from app.repositories.asset_repository import AssetRepository

msft_uuid = '79094066-a318-4727-9937-bfe30108132a'

def extract_metrics(info: dict) -> dict:
    return {
        # --- existing ones ---

        # Analyst consensus
        "recommendation_key":       info.get("recommendationKey"),       # "strong_buy", "buy", "hold"...
        "recommendation_mean":      info.get("recommendationMean"),      # 1.0 = strong buy, 5.0 = sell
        "nb_analyst_opinions":      info.get("numberOfAnalystOpinions"), # how many analysts (confidence weight)
        "target_mean":              info.get("targetMeanPrice"),
        "target_median":            info.get("targetMedianPrice"),
        "target_high":              info.get("targetHighPrice"),
        "target_low":               info.get("targetLowPrice"),
        "upside_to_target_pct":     (info.get("targetMeanPrice", 0) - info.get("currentPrice", 0)) / info.get("currentPrice", 1),

        # Momentum
        "52w_change_pct":           info.get("fiftyTwoWeekChangePercent"), # compare to a year ago
        "52w_from_high_pct":        info.get("fiftyTwoWeekHighChangePercent"),
        "52w_from_low_pct":         info.get("fiftyTwoWeekLowChangePercent"),
        "50d_avg_change_pct":       info.get("fiftyDayAverageChangePercent"),
        "200d_avg_change_pct":      info.get("twoHundredDayAverageChangePercent"),
        "beta":                     info.get("beta"),                               # volatility vs market

        # vs S&P
        "sp500_52w_change":         info.get("SandP52WeekChange"), # YoY s&p perf
        "relative_52w_perf":        info.get("fiftyTwoWeekChangePercent", 0) - info.get("SandP52WeekChange", 0),  # alpha

        # Ownership (smart money signal)
        "held_by_institutions":     info.get("heldPercentInstitutions"),  # institutional confidence ( 75 good )
        "held_by_insiders":         info.get("heldPercentInsiders"),      # no insider conviction ( < 5 ok )

        # Short interest (bearish signal)
        "short_ratio":              info.get("shortRatio"),               # take x days for short sellers to close positions ( 2 ok 5 bad )
        "short_pct_float":          info.get("shortPercentOfFloat"),      # 1.07% → only 1.07% of tradeable shares are being shorted ( 1 ok 5 bad)

        # Liquidity
        "current_ratio":            info.get("currentRatio"), # Current Assets / Current Liabilities  ( below 1 good else starts to get risky )
        "quick_ratio":              info.get("quickRatio"), # (Cash + Receivables) / Current Liabilities ( above 1.5 good,n 1 fine, 0.7 danger)

        # Governance/Risk ( Maybe to delete )
        "overall_risk":             info.get("overallRisk"),
        "audit_risk":               info.get("auditRisk"),
        "board_risk":               info.get("boardRisk"),             
        "compensation_risk":        info.get("compensationRisk"),  
        "shareholder_rights_risk":  info.get("shareHolderRightsRisk"),

        # Dividends ( maybe to delete ( payout_ratio is good ))
        "dividend_yield":           info.get("dividendYield"),
        "payout_ratio":             info.get("payoutRatio"),  # amount paid
        "five_year_avg_div_yield":  info.get("fiveYearAvgDividendYield"), # consitency check
    }

def compute_valuation_metrics(financials_at_date, balance_at_date, asset_price_at_date):

    price = asset_price_at_date.asset_price
    shares = balance_at_date["Ordinary Shares Number"]

    # Market cap
    market_cap = price * shares

    # P/E
    eps = financials_at_date["Diluted EPS"]
    pe_ratio = price / eps if eps else None

    # Price / Sales
    total_revenue = financials_at_date["Total Revenue"]
    price_to_sales = market_cap / total_revenue if total_revenue else None

    # EBITDA
    ebitda = financials_at_date["EBITDA"]

    # Enterprise Value
    total_debt = balance_at_date["Total Debt"]
    cash = balance_at_date["Cash And Cash Equivalents"]
    ev = market_cap + total_debt - cash

    # EV multiples
    ev_to_sales = ev / total_revenue if total_revenue else None
    ev_to_ebitda = ev / ebitda if ebitda else None

    # Book value / P-B
    common_equity = balance_at_date["Common Stock Equity"]
    book_value_per_share = common_equity / shares if shares else None
    price_to_book = price / book_value_per_share if book_value_per_share else None

    return {
        "market_cap": market_cap,
        "eps" : eps,
        "pe_ratio": pe_ratio,
        "price_to_sales": price_to_sales,
        "ev": ev,
        "ev_to_sales": ev_to_sales,
        "ebitda" : ebitda,
        "ev_to_ebitda": ev_to_ebitda,
        "price_to_book": price_to_book,
        "book_value_per_share": book_value_per_share,
    }

def extract_balance_sheet_metrics(balance_sheet_at_date, ebitda) -> dict:

    # Debt quality
    total_debt          = balance_sheet_at_date.get("Total Debt")
    long_term_debt      = balance_sheet_at_date.get("Long Term Debt")
    net_debt            = balance_sheet_at_date.get("Net Debt")
    common_equity       = balance_sheet_at_date.get("Common Stock Equity")

    return {
        # Debt structure
        "net_debt_to_ebitda":       net_debt/ebitda,
        "debt_to_equity":           total_debt / common_equity if common_equity else None,
        "lt_vs_total_debt_ratio":   long_term_debt / total_debt if total_debt else None,  # high = debt is long term = safer

        # Asset quality
        "goodwill":                 balance_sheet_at_date.get("Goodwill"),
        "goodwill_to_equity":       balance_sheet_at_date.get("Goodwill") / common_equity if common_equity else None,  # high = risky acquisitions
        "tangible_book_value":      balance_sheet_at_date.get("Tangible Book Value"),  # equity without goodwill

        # Growth signals (need multiple years)
        "total_assets":             balance_sheet_at_date.get("Total Assets"),
        "retained_earnings":        balance_sheet_at_date.get("Retained Earnings"),    # growing = profitable & reinvesting
        "invested_capital":         balance_sheet_at_date.get("Invested Capital"),
        "working_capital":          balance_sheet_at_date.get("Working Capital"),      # positive = healthy operations

        # Cash
        "cash":                     balance_sheet_at_date.get("Cash And Cash Equivalents"),
        "total_cash_and_st":        balance_sheet_at_date.get("Cash Cash Equivalents And Short Term Investments"),

        # PPE (capital intensity)
        "net_ppe":                  balance_sheet_at_date.get("Net PPE"),              # rising fast = heavy capex cycle
        "gross_ppe":                balance_sheet_at_date.get("Gross PPE"),
    }

async def main():
  asset = await AssetRepository().get_asset(msft_uuid)
  asset_prices = await AssetPriceRepository().get_price_of_one_asset(asset.uuid)
  t = yf.Ticker(asset.ticker_name)
  info = t.info # info now
  financials = t.financials
  balance_sheet = t.balance_sheet
  cahsflow = t.cashflow

  pd.set_option('display.max_rows', None)
  pprint.pprint(cahsflow)

  for date in financials.columns:
    target = pd.to_datetime(date).tz_localize(None)  # strip tz
    candidates = [
        item for item in asset_prices
        if abs((pd.to_datetime(item.asset_price_date).tz_localize(None) - target).days) <= 31 ]
    
    asset_price_at_date = min(
        candidates,
        key=lambda item: abs((pd.to_datetime(item.asset_price_date).tz_localize(None) - target).days),
        default=None )
    
    if(asset_price_at_date == None) :
       next

    financials_at_date = financials[date]
    balance_sheet_at_date = balance_sheet[date]

    metrics_histo = compute_valuation_metrics( financials_at_date, balance_sheet_at_date, asset_price_at_date)
    balance_sheet_histo = extract_balance_sheet_metrics( balance_sheet_at_date, metrics_histo['ebitda'])
    # Market cap, pe, cours/ventes, cours registre comptable, entreprise value/chiffre d'affaire, valeur entreprise / EBITDA$
    #print(metrics_histo)
    #pprint.pprint(balance_sheet_histo)

  info_current_year = extract_metrics(info) # all current info about stocks, above its history
  #pprint.pprint(metrics)

if __name__ == "__main__":
    asyncio.run(main())