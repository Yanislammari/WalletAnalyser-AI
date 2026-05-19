import numpy as np

def winsorize(df, cols, p=0.01):
    df = df.copy()
    for col in cols:
        lower = df[col].quantile(p)
        upper = df[col].quantile(1 - p)
        df[col] = np.clip(df[col], lower, upper)
    return df

def winsorize_two_sides(df, cols, min, max):
  df = df.copy()
  for col in cols:
      lower = df[col].quantile(min)
      upper = df[col].quantile(1 - max)
      df[col] = np.clip(df[col], lower, upper)
  return df

def classify_net_debt_ebitda(net_debt_ebitda):
    if net_debt_ebitda < 1:
        return 0
    elif 1 <= net_debt_ebitda < 2:
        return 1
    elif 2 <= net_debt_ebitda < 3:
        return 2
    elif 3 <= net_debt_ebitda < 5:
        return 3
    else:
        return 4