def backtest(df, initial_capital=10000):
  df['Position'] = df['Signal'].shift(1)
  df['Daily Return'] = df['Close'].pct_change()
  df['Strategy Return'] = df['Position'] * df['Daily Return']
  df['Cumulative Strategy Returns'] = (
      1 + df['Strategy Return']).cumprod() * initial_capital
  return df
