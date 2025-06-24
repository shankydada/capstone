import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import Adam


def prepare_lstm_data(series, window_size=10):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def predict_stock_price_all_models(symbol, start, end, model_type):
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None

        close_prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)

        if model_type == "Random Forest":
            X = scaled[:-1]
            y = scaled[1:]
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y.ravel())
            last_input = scaled[-1].reshape(1, -1)
            preds = [model.predict(last_input)[0]]
            for _ in range(6):
                last_input = scaler.transform([[preds[-1]]])
                preds.append(model.predict(last_input)[0])
            final_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()

        elif model_type in ["LSTM", "RNN", "DNN"]:
            X, y = prepare_lstm_data(scaled.flatten())
            if model_type != "DNN":
                X = X.reshape((X.shape[0], X.shape[1], 1))
            else:
                X = X.reshape(X.shape[0], X.shape[1])

            model = Sequential()
            if model_type == "LSTM":
                model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
            elif model_type == "RNN":
                model.add(SimpleRNN(50, return_sequences=False, input_shape=(X.shape[1], 1)))
            else:
                model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
                model.add(Dense(32, activation='relu'))

            model.add(Dense(1))
            model.compile(optimizer=Adam(0.001), loss='mse')
            model.fit(X, y, epochs=50, verbose=0)

            last_seq = scaled[-10:].reshape(1, 10, 1) if model_type != "DNN" else scaled[-10:].reshape(1, 10)
            preds = []
            for _ in range(7):
                pred = model.predict(last_seq, verbose=0)[0][0]
                preds.append(pred)
                new_input = np.append(last_seq.flatten()[1:], pred).reshape(1, 10)
                last_seq = new_input.reshape(1, 10, 1) if model_type != "DNN" else new_input

            final_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()

        future_dates = [df.index[-1] + timedelta(days=i + 1) for i in range(7)]
        prediction_df = pd.DataFrame({'Predicted Price': final_preds}, index=future_dates)
        return prediction_df

    except Exception as e:
        print(f"Prediction Error ({model_type}):", e)
        return None
