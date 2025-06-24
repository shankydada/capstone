from backend.indicators import calculate_rsi
from backend.indicators import calculate_sma


def sma_crossover_with_risk_control(df, stop_loss=-0.10, take_profit=0.15):
  df = calculate_sma(df, 50)
  df = calculate_sma(df, 200)
  df['Signal'] = 0
  position = 0
  buy_price = 0

  for i in range(1, len(df)):
    if df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i] and position == 0:
      df.at[df.index[i], 'Signal'] = 1  # Buy
      position = 1
      buy_price = df['Close'].iloc[i]

    elif position == 1:
      price_now = df['Close'].iloc[i]
      change_pct = (price_now - buy_price) / buy_price

      if (df['SMA_50'].iloc[i] < df['SMA_200'].iloc[i]) or (
          change_pct <= stop_loss) or (change_pct >= take_profit):
        df.at[df.index[i], 'Signal'] = -1  # Sell
        position = 0
        buy_price = 0

  return df


def rsi_strategy(df, lower=30, upper=70):
  df = calculate_rsi(df)
  df['Signal'] = 0
  df.loc[df['RSI'] < lower, 'Signal'] = 1  # Buy
  df.loc[df['RSI'] > upper, 'Signal'] = -1  # Sell
  return df

def macd_strategy(df):
  df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
  df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
  df['MACD'] = df['EMA_12'] - df['EMA_26']
  df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

  df['Signal'] = 0
  df.loc[df['MACD'] > df['Signal_Line'], 'Signal'] = 1
  df.loc[df['MACD'] < df['Signal_Line'], 'Signal'] = -1
  return df

def mean_reversion_strategy(df, window=20, threshold=0.05):
  df['SMA'] = df['Close'].rolling(window).mean()
  df['ZScore'] = (df['Close'] - df['SMA']) / df['Close'].rolling(window).std()

  df['Signal'] = 0
  df.loc[df['ZScore'] < -threshold, 'Signal'] = 1   # Buy
  df.loc[df['ZScore'] > threshold, 'Signal'] = -1   # Sell
  return df


