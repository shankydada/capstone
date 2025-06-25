# backend/ai_prediction.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Dropout
from keras.optimizers import Adam

def fetch_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        df = df[['Close']].dropna()
        return df
    except Exception as e:
        print("Data fetch error:", e)
        return None

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_dl_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
    elif model_type == 'RNN':
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=50))
    elif model_type == 'DNN':
        model.add(Dense(units=128, activation='relu', input_shape=(input_shape[0],)))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model

def predict_stock_price_all_models(symbol, start_date, end_date, model_type, forecast_days=6):
    df = fetch_data(symbol, start_date, end_date)
    if df is None or df.empty:
        print("No data fetched.")
        return None

    try:
        data, scaler = preprocess_data(df)
        window = 60
        X, y = create_sequences(data, window)

        print(f"Selected model: {model_type}, training on {X.shape[0]} samples.")

        if model_type in ['LSTM', 'RNN']:
            if len(X.shape) == 2:
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            model = build_dl_model(model_type, (X.shape[1], 1))
            print(f"Training {model_type} model...")
            model.fit(X, y, epochs=5, batch_size=32, verbose=1)
            print(f"{model_type} model training complete.\n")
        elif model_type == 'DNN':
            model = build_dl_model(model_type, (X.shape[1],))
            print("Training DNN model...")
            model.fit(X, y, epochs=5, batch_size=32, verbose=1)
            print("DNN model training complete.\n")
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100)
            print("Training Random Forest model...")
            model.fit(X, y)
            print("Random Forest training complete.\n")
        elif model_type == 'Linear Regression':
            model = LinearRegression()
            print("Training Linear Regression model...")
            model.fit(X, y)
            print("Linear Regression training complete.\n")
        else:
            print("Unsupported model type.")
            return None

        # Forecasting
        last_sequence = data[-window:]
        if last_sequence.shape[0] != window:
            print("Warning: Last sequence length mismatch.")
            return None

        future_predictions = []
        for _ in range(forecast_days):
            if model_type in ['LSTM', 'RNN']:
                input_seq = last_sequence.reshape(1, window, 1)
                pred = model.predict(input_seq, verbose=0)[0][0]
            elif model_type == 'DNN':
                input_seq = last_sequence.reshape(1, window)
                pred = model.predict(input_seq, verbose=0)[0][0]
            else:
                input_seq = last_sequence.reshape(1, -1)
                pred = model.predict(input_seq)[0]

            future_predictions.append(pred)
            last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

        predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        pred_df = pd.DataFrame({'Predicted Price': predicted_prices}, index=future_dates)
        return pred_df

    except Exception as e:
        print(f"Prediction Error ({model_type}):", e)
        return None
