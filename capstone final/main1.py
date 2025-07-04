import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime
from backend.data_loader import load_data
from backend.strategy import (
    sma_crossover_with_risk_control, rsi_strategy,
    macd_strategy, mean_reversion_strategy
)
from backend.backtesting import backtest
from backend.metrices import calculate_sharpe_ratio, calculate_max_drawdown
from backend.ai_prediction import predict_stock_price_all_models
from backend.alert import get_past_alerts, get_future_alerts

from backend.multi_timeframe import (
    load_multi_timeframe_data,
    prepare_daily_data,
    prepare_hourly_data,
    run_multi_timeframe_backtest,
    calculate_backtest_metrics
)

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
     background-image: url("D:\ATIK\Desktop\capstone\trading-bearish-1000x484.jpg");
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
/* ===== GLOBAL INPUT STYLING ===== */     
/* All text input labels */
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] input {
    font-size: 18px !important;
    color: #FFFFFF !important;
}

/* All date input labels */
div[data-testid="stDateInput"] label,
div[data-testid="stDateInput"] input {
    font-size: 18px !important;
    color: #FFFFFF !important;
}

/* All slider labels */
div[data-testid="stSlider"] label {
    font-size: 18px !important;
    color: #FFFFFF !important;
}

/* All select/dropdown labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] div {
    font-size: 18px !important;
    color: #FFFFFF !important;
}

/* All number input labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] input {
    font-size: 18px !important;
    color: #FFFFFF !important;  /* Medium purple */
}

/* Button text */
div[data-testid="stButton"] button p {
    font-size: 18px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
    color: #FFFFFF !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-size: 30px !important;
    color: #FFFFFF !important;  /* Spring green */
}
            
/* Full background image */
body {
    background-image: url('https://4kwallpjpgapers.com/images/walls/thumbs/13833.');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
}

/* Vibrant Neon Tabs */
.css-1r6slb0 { background-color: rgba(0, 0, 0, 0.7) !important; }

/* Target Streamlit headers correctly */
h1, h2, h3, h4,
[data-testid="stHeader"] h1,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    font-family: 'Segoe UI', sans-serif !important;
    color: #ffffff !important;
    text-shadow: 0 0 10px #000000, 0 0 10px #000000;
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
    box-shadow: 0 0 28px #000000, 0 0 28px #000000;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Layout Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5  = st.tabs(["📈 BACKTESTING", "🤖 AI PREDICTOR", "📊 VISUAL INSIGHTS", "📈 MULTI TIMEFRAME SIGNAL ANALYSIS" , "ALERT"])

st.markdown("""
    <style>
    /* Base styling for all tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }

    /* Tab 1 (📈 BACKTESTING) */
    .stTabs [data-baseweb="tab"]:nth-child(1) {
        color: white;
        background-color: #1e1e1e;
    }

    /* Tab 2 (🤖 AI PREDICTOR) */
    .stTabs [data-baseweb="tab"]:nth-child(2) {
        color: #00ffe1;
        background-color: #0a0a0a;
    }

    /* Tab 3 (📊 VISUAL INSIGHTS) */
    .stTabs [data-baseweb="tab"]:nth-child(3) {
        color: #ffc107;
        background-color: #111111;
    }
          
              /* Tab 4 (📈 MULTI TIMEFRAME SIGNAL ANALYSIS) */
    .stTabs [data-baseweb="tab"]:nth-child(4) {
        color: #00FF00;
        background-color: #111111;
    }
            
                  
              /* Tab 5 (📈 ALERT ) */
    .stTabs [data-baseweb="tab"]:nth-child(5) {
        color: #FFFF00;
        background-color: #111111;
    }

    /* 🔴 Selected (clicked) tab — applies to the active one */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: red !important;
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 0 12px red;
    }
    </style>
""", unsafe_allow_html=True)


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

            
# ----------------------
# Tab 4
# ----------------------
with tab4:
    st.title("📊 Multi-Timeframe Signal Analysis")
    
    # Parameters Section - Similar to AI Predictor layout
    with st.expander("⚙️ Strategy Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Stock Symbol", "AAPL", key="mtf_symbol")
            rsi_entry = st.slider("RSI Entry Threshold", 10, 40, 30, key="mtf_rsi_entry")
            
        with col2:
            daily_period = st.selectbox("Daily Period", ["3mo", "6mo", "1y", "2y"], index=2)
            rsi_exit = st.slider("RSI Exit Threshold", 60, 90, 70, key="mtf_rsi_exit")
            
        with col3:
            hourly_period = st.selectbox("Hourly Period", ["30d", "60d", "90d", "180d"], index=1)
            holding_period = st.slider("Holding Period (hours)", 1, 24, 8, key="mtf_holding")
    
    # Run Button - Centered below parameters
    st.markdown("""
    <style>
    .centered-button {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    run_analysis = st.button("🚀 Run Multi-Timeframe Analysis", key="mtf_run")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if run_analysis:
        with st.spinner("Running multi-timeframe analysis..."):
            try:
                # Load and prepare data
                df_daily, df_hourly = load_multi_timeframe_data(symbol, daily_period, hourly_period)
                df_daily = prepare_daily_data(df_daily)
                df_hourly = prepare_hourly_data(df_hourly, df_daily)
                
                # Run backtest
                results = run_multi_timeframe_backtest(
                    df_hourly, 
                    holding_period=holding_period,
                    rsi_entry=rsi_entry,
                    rsi_exit=rsi_exit
                )
                
                # Get trades and metrics
                backtest_df = results[results['position'] == 1]
                metrics = calculate_backtest_metrics(backtest_df)
                
                # Store in session state
                st.session_state.mtf_results = {
                    'hourly_data': results,
                    'backtest_df': backtest_df,
                    'metrics': metrics,
                    'params': {
                        'symbol': symbol,
                        'rsi_entry': rsi_entry,
                        'rsi_exit': rsi_exit,
                        'holding_period': holding_period
                    }
                }
                
                st.success("Analysis completed!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Results Display - Below parameters like AI Predictor
    if 'mtf_results' in st.session_state:
        results = st.session_state.mtf_results
        metrics = results['metrics']
        backtest_df = results['backtest_df']
        hourly_data = results['hourly_data']
        params = results['params']
        
        # Metrics Cards
        st.subheader(f"📊 Performance Metrics for {params['symbol']}")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Trades", metrics['total_trades'])
        with cols[1]:
            st.metric("Total PnL", f"${metrics['total_pnl']:.2f}")
        with cols[2]:
            st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
        with cols[3]:
            st.metric("Avg PnL/Trade", f"${metrics['avg_pnl']:.2f}")
        
        # Charts Section
        st.subheader("📈 Price & Signals")
        
        # Price with Signals Chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hourly_data['Datetime'], 
            y=hourly_data['Close'], 
            name='Price',
            line=dict(color='#00FFAA', width=2)
        ))
        
        if len(backtest_df) > 0:
            fig_price.add_trace(go.Scatter(
                x=backtest_df['entry_time'],
                y=backtest_df['entry_price'],
                mode='markers',
                marker=dict(color='#00FF00', size=10, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey')),
                name='Buy Signal'
            ))

            fig_price.add_trace(go.Scatter(
                x=backtest_df['exit_time'],
                y=backtest_df['exit_price'],
                mode='markers',
                marker=dict(color='#FF0000', size=10, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey')),
                name='Sell Signal'
            ))
        
        fig_price.update_layout(
            template="plotly_dark",
            title=f"{params['symbol']} Price with Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # RSI Chart
        st.subheader("📉 RSI with Signal Levels")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=hourly_data['Datetime'], 
            y=hourly_data['rsi'], 
            name='RSI',
            line=dict(color='#1E90FF', width=2)
        ))
        fig_rsi.add_hline(
            y=params['rsi_entry'], 
            line_dash="dash", 
            line_color="#00FF00",
            annotation_text=f"Entry ({params['rsi_entry']})", 
            annotation_position="bottom right"
        )
        fig_rsi.add_hline(
            y=params['rsi_exit'], 
            line_dash="dash", 
            line_color="#FF0000",
            annotation_text=f"Exit ({params['rsi_exit']})", 
            annotation_position="top right"
        )
        fig_rsi.update_layout(
            template="plotly_dark",
            title="RSI with Trading Levels",
            xaxis_title="Date",
            yaxis_title="RSI",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Trade Details
        if len(backtest_df) > 0:
            st.subheader("📋 Trade Details")
            st.dataframe(
                backtest_df[[
                    'entry_time', 'exit_time', 
                    'entry_price', 'exit_price',
                    'pnl', 'duration'
                ]].sort_values('entry_time', ascending=False),
                height=400,
                use_container_width=True
            )
            
            # Download button
            csv = backtest_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trade Data",
                data=csv,
                file_name=f"{params['symbol']}_multi_timeframe_trades.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No trades were executed with current parameters")

# ---------------------------
# Tab 5:ALERT
# ---------------------------

with tab5:
    st.title("🔔 Stock Alert System")
    
    with st.sidebar:
        st.header("⚙ Alert Controls")
        alert_symbol = st.text_input("Stock Symbol", "AAPL", key="alert_sym")
        alert_start = st.date_input("Start Date", datetime(2023, 1, 1), key="alert_start")
        alert_end = st.date_input("End Date", datetime.today(), key="alert_end")
        hist_threshold = st.slider("Historical Volatility Threshold", 1.0, 5.0, 2.0, 0.5)
        future_threshold = st.slider("Future Change Threshold (%)", 1, 20, 5)
        run_alerts = st.button("🔔 Generate Alerts")
    
    if run_alerts:
        # Load data
        df = load_data(alert_symbol, alert_start, alert_end)
        
        if df is None or df.empty:
            st.error("Failed to load data")
        else:
            # Historical alerts
            st.subheader("⚠ Historical Price Alerts")
            hist_alerts = get_past_alerts(df, threshold=hist_threshold)
            
            if not hist_alerts.empty:
                st.dataframe(hist_alerts.style.applymap(
                    lambda x: 'color: red' if x < 0 else 'color: green', 
                    subset=['Pct_Change']
                ))
                
                # Plot with alert markers
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                fig.add_trace(go.Scatter(
                    x=hist_alerts.index,
                    y=hist_alerts['Close'],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Down Alert'
                ))
                fig.add_trace(go.Scatter(
                    x=hist_alerts[hist_alerts['Pct_Change'] > 0].index,
                    y=hist_alerts[hist_alerts['Pct_Change'] > 0]['Close'],
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='Up Alert'
                ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant historical movements detected")
            
            # Future alerts
            st.subheader("🔮 Future Price Alerts")
            current_price = df['Close'].iloc[-1]
            
            # Get prediction (reuse existing AI model)
            prediction_df = predict_stock_price_all_models(
                alert_symbol, 
                alert_start, 
                alert_end,
                "LSTM",
                5  # Forecast 5 days
            )
            
            if prediction_df is not None:
                alert_msg = get_future_alerts(
                    prediction_df['Predicted Price'],
                    current_price,
                    future_threshold/100
                )
                
                st.subheader(alert_msg)
                
                # Plot prediction with alert zone
                fig = go.Figure()
