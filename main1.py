# Import Required Libraries
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objs as go
import pandas as pd
from backend.data_loader import load_data
from backend.strategy import (
    sma_crossover_with_risk_control,
    rsi_strategy,
    macd_strategy,
    mean_reversion_strategy
)
from backend.backtesting import backtest
from backend.metrices import calculate_sharpe_ratio, calculate_max_drawdown
from backend.ai_prediction import predict_stock_price_all_models  # Updated function

# Page Config
st.set_page_config(page_title="🚀 AI Stock Backtester", layout="wide", page_icon="📊")

# Custom CSS with dark/light themes
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #004e92, #000428);
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFDD00;
        text-shadow: 2px 2px 4px #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Tabs Layout
tab1, tab2 = st.tabs(["📈 Backtesting", "🤖 AI Predictor"])

# Tab 1: Backtesting
with tab1:
    st.markdown("""
    <h1 style='text-align: center;'>📈 Advanced Stock Strategy Backtester</h1>
    <hr style='border-top: 3px solid #bbb;'>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Strategy Settings")
        symbol = st.text_input("Stock Symbol", "AAPL")
        start = st.date_input("Start Date")
        end = st.date_input("End Date")
        strategy_option = st.selectbox("Strategy", [
            "SMA Crossover", "RSI", "MACD", "Mean Reversion"])
        stop_loss = st.slider("Stop-Loss (%)", 1, 20, 10) / 100
        take_profit = st.slider("Take-Profit (%)", 1, 30, 15) / 100
        run = st.button("🚀 Run Backtest")

    if run:
        st.subheader("📊 Results")
        df = load_data(symbol, start, end)

        if df is None or df.empty:
            st.error("❌ Failed to load stock data.")
        else:
            if strategy_option == "SMA Crossover":
                df = sma_crossover_with_risk_control(df, stop_loss, take_profit)
            elif strategy_option == "RSI":
                df = rsi_strategy(df)
            elif strategy_option == "MACD":
                df = macd_strategy(df)
            elif strategy_option == "Mean Reversion":
                df = mean_reversion_strategy(df)

            df = backtest(df)
            st.write("### Signal Data")
            st.dataframe(df.tail(), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            if strategy_option == "SMA Crossover":
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200'))
            elif strategy_option == "MACD":
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'))
            elif strategy_option == "RSI":
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Strategy Performance")
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative Strategy Returns'], name='Cumulative Returns', line=dict(color='green')))
            st.plotly_chart(perf_fig, use_container_width=True)

            sharpe = calculate_sharpe_ratio(df)
            drawdown = calculate_max_drawdown(df)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Max Drawdown", f"{drawdown:.2%}")

# Tab 2: AI Predictor
with tab2:
    st.markdown("""
    <h1 style='text-align: center;'>🤖 Multi-Model AI Stock Predictor</h1>
    <hr style='border-top: 3px solid #bbb;'>
    """, unsafe_allow_html=True)

    symbol_pred = st.text_input("Enter stock symbol for prediction", "AAPL")
    start_pred = st.date_input("Start Date for Training")
    end_pred = st.date_input("End Date for Training")
    model_type = st.selectbox("Select AI Model", ["Random Forest", "LSTM", "RNN", "DNN"])
    predict = st.button("Predict Future Prices")

    if predict:
        prediction_df = predict_stock_price_all_models(symbol_pred, start_pred, end_pred, model_type)
        if prediction_df is not None:
            st.success(f"✅ Prediction using {model_type} completed!")
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Predicted Price'], name='Predicted Price', line=dict(color='orange')))
            st.plotly_chart(fig_pred, use_container_width=True)
            st.dataframe(prediction_df)
        else:
            st.error("❌ AI Model failed to generate prediction.")