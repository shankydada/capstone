import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

def load_multi_timeframe_data(symbol, daily_period="1y", hourly_period="60d"):
    """Load daily and hourly data for given symbol"""
    try:
        df_daily = yf.download(symbol, interval="1d", period=daily_period, auto_adjust=True)
        df_hourly = yf.download(symbol, interval="1h", period=hourly_period, auto_adjust=True)
        
        # Flatten multi-index columns if they exist
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)
        if isinstance(df_hourly.columns, pd.MultiIndex):
            df_hourly.columns = df_hourly.columns.get_level_values(0)
            
        return df_daily, df_hourly
    except Exception as e:
        raise Exception(f"Error downloading data: {e}")

def prepare_daily_data(df_daily):
    """Calculate daily indicators"""
    df_daily = df_daily[['Close']].copy()
    df_daily.index.name = 'Date'
    df_daily = df_daily.asfreq('D')
    df_daily['Close'] = df_daily['Close'].ffill()
    
    # Calculate SMAs
    close_daily_series = df_daily['Close'].squeeze()
    df_daily['sma_50'] = SMAIndicator(close=close_daily_series, window=50).sma_indicator()
    df_daily['sma_200'] = SMAIndicator(close=close_daily_series, window=200).sma_indicator()
    df_daily['trend_up'] = df_daily['sma_50'] > df_daily['sma_200']
    
    # Prepare for merge
    df_daily.reset_index(inplace=True)
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.date
    
    return df_daily

def prepare_hourly_data(df_hourly, df_daily):
    """Calculate hourly indicators and merge with daily data"""
    df_hourly = df_hourly.copy()
    df_hourly.reset_index(inplace=True)
    df_hourly['Datetime'] = pd.to_datetime(df_hourly['Datetime'])
    df_hourly['Date'] = df_hourly['Datetime'].dt.date
    df_hourly['Close'] = df_hourly['Close'].astype(float).ffill()
    
    # Merge with daily trend
    df_hourly = df_hourly.merge(df_daily[['Date', 'trend_up']], on='Date', how='left')
    df_hourly = df_hourly.sort_values('Datetime')
    
    # Calculate indicators
    close_hourly_series = df_hourly['Close'].squeeze()
    df_hourly['rsi'] = RSIIndicator(close=close_hourly_series, window=14).rsi()
    df_hourly['ema_20'] = EMAIndicator(close=close_hourly_series, window=20).ema_indicator()
    
    return df_hourly

def run_multi_timeframe_backtest(df_hourly, holding_period=8, rsi_entry=30, rsi_exit=70):
    """Run the backtest with given parameters"""
    df = df_hourly.copy()
    
    # Define signals
    df['entry_signal'] = (df['rsi'] < rsi_entry)  # & (df['trend_up']) & (df['Close'] > df['ema_20'])
    df['exit_signal'] = (df['rsi'] > rsi_exit)
    
    # Initialize backtest columns
    df['position'] = 0
    df['entry_price'] = None
    df['exit_price'] = None
    df['pnl'] = None
    df['entry_time'] = None
    df['exit_time'] = None
    df['duration'] = None
    
    # Execute backtest
    last_exit_idx = -1
    for idx in df.index:
        if df.loc[idx, 'entry_signal'] and idx > last_exit_idx:
            entry_price = df.loc[idx, 'Close']
            entry_time = df.loc[idx, 'Datetime']
            
            # Find exit point
            lookahead = df.iloc[idx+1:idx+1+holding_period]
            exit_idx = None
            
            # Check for exit signals first
            exit_candidates = lookahead[lookahead['exit_signal']]
            if not exit_candidates.empty:
                exit_idx = exit_candidates.index[0]
            
            # If no exit signal, use end of holding period
            if exit_idx is None and not lookahead.empty:
                exit_idx = lookahead.index[-1]
            
            if exit_idx is not None:
                exit_price = df.loc[exit_idx, 'Close']
                df.at[idx, 'position'] = 1
                df.at[idx, 'entry_price'] = entry_price
                df.at[idx, 'exit_price'] = exit_price
                df.at[idx, 'entry_time'] = entry_time
                df.at[idx, 'exit_time'] = df.loc[exit_idx, 'Datetime']
                df.at[idx, 'pnl'] = exit_price - entry_price
                df.at[idx, 'duration'] = (df.loc[exit_idx, 'Datetime'] - entry_time).total_seconds()/3600
                last_exit_idx = exit_idx
    
    return df

def calculate_backtest_metrics(backtest_df):
    """Calculate performance metrics from backtest results"""
    if len(backtest_df) == 0:
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'win_rate': 0,
            'max_win': 0,
            'max_loss': 0
        }
    
    total_pnl = backtest_df['pnl'].sum()
    avg_pnl = backtest_df['pnl'].mean()
    win_rate = (backtest_df['pnl'] > 0).mean()
    max_win = backtest_df['pnl'].max()
    max_loss = backtest_df['pnl'].min()
    
    return {
        'total_trades': len(backtest_df),
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'max_win': max_win,
        'max_loss': max_loss
    }