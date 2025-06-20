import streamlit as st
from backend.data_loader import load_data
from backend.strategy import sma_crossover_strategy
from backend.backtesting import backtest
from backend.metrices import calculate_sharpe_ratio, calculate_max_drawdown

st.title("📊 Stock Backtesting App")

symbol = st.text_input("Enter stock symbol", "AAPL")
start = st.date_input("Start Date")
end = st.date_input("End Date")

if st.button("Run Backtest"):
    df = load_data(symbol, start, end)
    df = sma_crossover_strategy(df)
    df = backtest(df)

    st.write("### Strategy Signal Data", df.tail())
    st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])
    st.line_chart(df['Cumulative Strategy Returns'])

    sharpe = calculate_sharpe_ratio(df)
    drawdown = calculate_max_drawdown(df)

    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("Max Drawdown", f"{drawdown:.2%}")
