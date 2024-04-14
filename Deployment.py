%%writefile streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def download_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)
    return data

# Function to calculate daily average
def calculate_daily_average(data):
    data['Daily Average'] = (data['High'] + data['Low']) / 2

# Function to calculate moving averages
def calculate_moving_averages(data, short_window=12, long_window=26):
    data['Short MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

# Function to calculate moving average crossover
def calculate_ma_crossover(data):
    data['MA Crossover'] = data['Short MA'] - data['Long MA']

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal=9):
    exp1 = data['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()

# Function to calculate Bollinger Bandwidth
def calculate_bollinger_bandwidth(data, window=20):
    sma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    data['Bollinger Bandwidth'] = (sma + 2 * std) - (sma - 2 * std)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

# Function to calculate Smoothed Stochastic Oscillator
def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    data['%K'] = (data['Close'] - low_min) * 100 / (high_max - low_min)
    data['%D'] = data['%K'].rolling(window=d_window).mean()

# Function to calculate Volatility
def calculate_volatility(data, window=7):
    data['Volatility'] = data['Close'].pct_change().rolling(window=window).std() * (252 ** 0.5)

# Function to add calculated columns to the DataFrame
def add_calculated_columns(data):
    calculate_daily_average(data)
    calculate_moving_averages(data)
    calculate_ma_crossover(data)
    calculate_macd(data)
    calculate_bollinger_bandwidth(data)
    calculate_rsi(data)
    calculate_stochastic_oscillator(data)
    calculate_volatility(data)

def preprocess_data(data):
    data.fillna(0, inplace=True)

def cluster_data(data, n_clusters=3):
    clustering_features = ['Daily Average', 'MACD', 'RSI', 'Bollinger Bandwidth', 'Volatility']
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data[clustering_features].fillna(0))
    data['Trend Cluster'] = clusters

def classify_trends(data):
    features = ['Daily Average', 'Short MA', 'Long MA', 'MACD', 'RSI', '%K', '%D', 'Volatility']
    X = data[features].fillna(0)
    y = data['Trend Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifier = GradientBoostingClassifier(random_state=0)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, predictions))

    data['Predicted Trend'] = classifier.predict(X)

def predict_price(data, target_feature, symbol):
    features = ['Daily Average', 'Short MA', 'Long MA', 'MACD', 'RSI', '%K', '%D', 'Volatility']
    X = data[features].fillna(0)
    y = data[target_feature]

    model = BayesianRidge()
    model.fit(X, y)
    predictions = model.predict(X)

    data['Predicted ' + target_feature] = predictions

    print(f"Last 5 Predicted {target_feature} Prices for {symbol}:")
    print(data[data['Ticker'] == symbol][['Date', 'Predicted ' + target_feature]].tail())


def streamlit_app():
    st.title('Stock Trend and Price Prediction')

    # User input for stock selection and dates
    symbols = st.text_input("Enter stock symbols separated by space").split()
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
    target_feature = st.selectbox("Select the feature to predict", ["Open", "Close"])

    if st.button('Predict'):
        with st.spinner('Fetching data and predicting...'):
            # Fetch and process the stock data
            stock_data = download_stock_data(symbols, str(start_date), str(end_date))
            add_calculated_columns(stock_data)
            preprocess_data(stock_data)
            cluster_data(stock_data)
            classify_trends(stock_data)

            # Prediction
            symbol = symbols[0] if len(symbols) == 1 else symbols[0]  # Or let user choose
            predict_price(stock_data, target_feature, symbol)

            # Display results
            st.subheader(f"Predicted {target_feature} prices for {symbol}")
            st.write(stock_data[stock_data['Ticker'] == symbol][['Date', 'Predicted ' + target_feature]].tail())

            # Optionally display full data
            st.subheader("Full Data")
            st.write(stock_data)

if __name__ == '__main__':
    streamlit_app()
