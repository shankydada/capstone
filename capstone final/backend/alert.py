import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

def get_past_alerts(data, window=20, threshold=2.0):
    """Identify significant historical price movements"""
    data = data.copy()
    data['Pct_Change'] = data['Close'].pct_change() * 100
    data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    
    # Identify significant moves
    data['Major_Move'] = np.where(
        abs(data['Pct_Change']) > threshold * data['ATR'].rolling(window).mean(),
        data['Pct_Change'],
        np.nan
    )
    
    return data[data['Major_Move'].notna()][['Close', 'Pct_Change', 'Major_Move']]

def get_future_alerts(predicted_prices, current_price, threshold=0.05):
    """Predict significant future price movements"""
    max_change = (predicted_prices.max() - current_price) / current_price
    min_change = (predicted_prices.min() - current_price) / current_price
    
    if max_change > threshold or min_change < -threshold:
        direction = "increase" if max_change > abs(min_change) else "decrease"
        magnitude = max(max_change, abs(min_change)) * 100
        return f"ALERT: Predicted {direction} of {magnitude:.2f}% within forecast period"
    return "No significant movement predicted"