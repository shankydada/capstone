def calculate_sma(df, period=20):
  df[f"SMA_{period}"] = df['Close'].rolling(window=period).mean()
  return df


def calculate_rsi(df, period=14):
  delta = df['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
  rs = gain / loss
  df['RSI'] = 100 - (100 / (1 + rs))
  return df
