import numpy as np


def calculate_sharpe_ratio(df):
    return df['Strategy Return'].mean() / df['Strategy Return'].std(
    ) * np.sqrt(252)


def calculate_max_drawdown(df):
    cumulative = df['Cumulative Strategy Returns']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
