import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from backend.data_loader import load_data
from backend.strategy import (
    sma_crossover_with_risk_control, rsi_strategy,
    macd_strategy, mean_reversion_strategy
)
from backend.backtesting import backtest
from backend.metrices import calculate_sharpe_ratio, calculate_max_drawdown
from backend.ai_prediction import predict_stock_price_all_models

# Page config
page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://marketspotter.io/wp-content/uploads/2023/09/trading-bearish-1000x484.jpg");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    }

    [data-testid="stHeader"]{
    background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"]{
     background-image: url("https://th.bing.com/th/id/OIP.OBewGnRT9OghDc7TpUaUOQHaER?w=266&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3");
    background-position: center;
    }


    </style>"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.set_page_config(page_title="AI Stock Dashboard", layout="wide", page_icon="📊")

# ---------------------------
# Custom STYLING: Background, Inputs, Buttons, Tabs
# ---------------------------
st.markdown("""
<style>
/* Full background image */
body {
    background-image: url('https://4kwallpapers.com/images/walls/thumbs/13833.jpg');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
}

/* Vibrant Neon Tabs */
.css-1r6slb0 { background-color: rgba(0, 0, 0, 0.7) !important; }

h1, h2, h3, h4 {
    font-family: 'Segoe UI', sans-serif;
    color: #00ffe1;
    text-shadow: 0 0 8px #00fff7, 0 0 4px #00c3ff;
}

/* Sidebar Styling */
.css-1d391kg { background-color: rgba(0,0,0,0.8) !important; }
.css-1v0mbdj p {
    color: #ffffff;
}

/* Input fields & dropdowns */
input, select, textarea {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    border: 2px solid #00e1ff !important;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0px 0px 6px #00ffff;
}

/* Sliders */
.stSlider > div > div {
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(90deg, #00f2ff, #00ff85, #00c3ff);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1.4em;
    font-size: 16px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 0 10px #00ffe1, 0 0 20px #00ffa6;
}

.stButton > button:hover {
    transform: scale(1.07);
    background: linear-gradient(270deg, #00f2ff, #00ff85, #00c3ff);
    box-shadow: 0 0 18px #00ffe1, 0 0 28px #00ffa6;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Layout Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📈 Backtesting", "🤖 AI Predictor", "📊 Visual Insights"])

# ---------------------------
# Tab 1: Backtesting
# ---------------------------
with tab1:
    st.title("📈 Strategy Backtester")

    with st.sidebar:
        st.header("⚙️ Backtest Controls")
        symbol = st.text_input("Stock Symbol", "AAPL")
        start = st.date_input("Start Date", value=datetime(2022, 1, 1))
        end = st.date_input("End Date", value=datetime.today())
        strategy_option = st.selectbox("Strategy", ["SMA Crossover", "RSI", "MACD", "Mean Reversion"])
        stop_loss = st.slider("Stop Loss %", 1, 20, 10) / 100
        take_profit = st.slider("Take Profit %", 1, 30, 15) / 100
        run = st.button("🚀 Run Backtest")

    if run:
        df = load_data(symbol, start, end)
        if df is None or df.empty:
            st.error("Failed to load data.")
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
            triggered = df[df['Signal'] != 0]
            st.success("Strategy applied successfully!")
            st.subheader("🔍 Signal Data")
            st.dataframe(triggered if not triggered.empty else df, use_container_width=True)

            csv_download = (triggered if not triggered.empty else df).to_csv(index=True).encode('utf-8')
            st.download_button("📥 Download CSV", csv_download, "backtest_result.csv", "text/csv")

            st.subheader("📉 Price with Indicators")
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            if 'SMA_50' in df.columns:
                price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
            if 'SMA_200' in df.columns:
                price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200'))
            if 'MACD' in df.columns:
                price_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
            if 'Signal_Line' in df.columns:
                price_fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'))
            price_fig.update_layout(template="plotly_dark")
            st.plotly_chart(price_fig, use_container_width=True)

            st.subheader("📊 Strategy Performance")
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative Strategy Returns'],
                                          name='Cumulative Returns', line=dict(color='green')))
            perf_fig.update_layout(template="plotly_dark")
            st.plotly_chart(perf_fig, use_container_width=True)

            st.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(df):.2f}")
            st.metric("Max Drawdown", f"{calculate_max_drawdown(df):.2%}")

# ---------------------------
# Tab 2: AI Predictor
# ---------------------------
with tab2:
    st.title("🤖 AI Price Predictor")
    symbol_pred = st.text_input("Stock Symbol for Prediction", "AAPL", key="sym_pred")
    start_pred = st.date_input("Training Start Date", value=datetime(2022, 1, 1), key="start_pred")
    end_pred = st.date_input("Training End Date", value=datetime.today(), key="end_pred")
    model_type = st.selectbox("Model Type", ["Random Forest", "LSTM", "RNN", "DNN"], key="model")
    forecast_days = st.slider("Forecast Days", 1, 10, 6)

    if st.button("Run Prediction"):
        prediction_df = predict_stock_price_all_models(symbol_pred, start_pred, end_pred, model_type, forecast_days)
        if prediction_df is not None and not prediction_df.empty:
            st.success("Prediction completed!")
            st.dataframe(prediction_df)
            pred_fig = px.line(prediction_df, x=prediction_df.index, y='Predicted Price', title="📈 Predicted Future Prices")
            st.plotly_chart(pred_fig, use_container_width=True)
        else:
            st.error("Prediction failed or empty result.")

# ---------------------------
# Tab 3: Visual Insights
# ---------------------------
with tab3:
    st.title("📊 Visual Insights")

    vis_symbol = st.text_input("📌 Stock Symbol", "AAPL", key="vis_symbol")
    vis_start = st.date_input("Start Date", datetime(2022, 1, 1), key="vis_start")
    vis_end = st.date_input("End Date", datetime.today(), key="vis_end")
    run_vis = st.button("🔍 Show Insights")

    if run_vis:
        try:
            df = load_data(vis_symbol, vis_start, vis_end)

            if df is None or df.empty:
                st.error("No data found.")
            else:
                st.success(f"Showing visuals for {vis_symbol}")

                st.subheader("📋 Data Snapshot")
                st.dataframe(df.tail(), use_container_width=True)

                st.subheader("📈 Price Trends")
                core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in core_cols:
                    if col in df.columns:
                        fig = px.line(df, x=df.index, y=col, title=f"{col} over Time")
                        st.plotly_chart(fig, use_container_width=True)

                if 'Close' in df.columns:
                    df['Returns'] = df['Close'].pct_change()
                    st.subheader("📉 Risk–Reward Potential")
                    mean_return = df['Returns'].mean()
                    std_dev = df['Returns'].std()
                    sharpe = (mean_return / std_dev) * (252 ** 0.5) if std_dev != 0 else 0
                    cumulative = (1 + df['Returns'].fillna(0)).cumprod()
                    drawdown = (cumulative / cumulative.cummax() - 1).min()

                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    st.metric("Max Drawdown", f"{drawdown:.2%}")
                    st.metric("Volatility (Daily)", f"{std_dev:.2%}")
                    st.plotly_chart(px.histogram(df, x='Returns', nbins=60, title="Return Distribution"), use_container_width=True)

                if 'Close' in df.columns:
                    st.subheader("📊 Moving Averages")
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
                    fig.update_layout(template="plotly_dark", title="SMA Trends")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("🔎 Correlation Matrix")
                numeric = df.select_dtypes(include='number')
                if not numeric.empty:
                    st.plotly_chart(px.imshow(numeric.corr(), title="Correlation Heatmap"), use_container_width=True)
                else:
                    st.warning("No numeric columns found for correlation.")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
