import yfinance as yf
import pandas as pd


def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df
