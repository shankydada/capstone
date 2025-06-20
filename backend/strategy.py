from .utils import calculate_sma


def sma_crossover_strategy(df):
  df = calculate_sma(df, 50)
  df = calculate_sma(df, 200)
  df['Signal'] = 0
  df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = 1
  df.loc[df['SMA_50'] < df['SMA_200'], 'Signal'] = -1
  return df
