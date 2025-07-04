import streamlit as st
from backend.data_loader import load_data
from backend.strategy import (
    sma_crossover_with_risk_control,
    rsi_strategy,
    macd_strategy,
    mean_reversion_strategy
)
from backend.backtesting import backtest
from backend.metrices import calculate_sharpe_ratio, calculate_max_drawdown

st.set_page_config(page_title="Stock Backtesting App", layout="wide")
st.title("\U0001F4CA Stock Backtesting App")

symbol = st.text_input("Enter stock symbol", "AAPL")
start = st.date_input("Start Date")
end = st.date_input("End Date")

strategy_option = st.selectbox("Select Strategy", [
    "SMA Crossover", "RSI", "MACD", "Mean Reversion"])

stop_loss = st.slider("Stop-Loss (%)", min_value=1, max_value=20, value=10) / 100
take_profit = st.slider("Take-Profit (%)", min_value=1, max_value=30, value=15) / 100

if st.button("Run Backtest"):
    df = load_data(symbol, start, end)

    if df is None or df.empty:
        st.error("\u274C Failed to load stock data. Check the symbol and date range.")
    else:
        if strategy_option == "SMA Crossover":
            df = sma_crossover_with_risk_control(df, stop_loss, take_profit)
        elif strategy_option == "RSI":
            df = rsi_strategy(df)
        elif strategy_option == "MACD":
            df = macd_strategy(df)
        elif strategy_option == "Mean Reversion":
            df = mean_reversion_strategy(df)

        if df is not None and not df.empty:
            df = backtest(df)
            st.write("### Strategy Signal Data", df.tail())

            st.write("### Price Chart")
            if strategy_option == "SMA Crossover":
                st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])
            elif strategy_option == "MACD":
                st.line_chart(df[['Close', 'MACD', 'Signal_Line']])
            elif strategy_option == "RSI":
                st.line_chart(df[['Close', 'RSI']])
            else:
                st.line_chart(df[['Close']])

            st.write("### Portfolio Performance")
            st.line_chart(df['Cumulative Strategy Returns'])

            sharpe = calculate_sharpe_ratio(df)
            drawdown = calculate_max_drawdown(df)

            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Max Drawdown", f"{drawdown:.2%}")
        else:
            st.error("\u274C Strategy returned empty results.")

