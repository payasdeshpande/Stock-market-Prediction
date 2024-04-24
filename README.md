### Stock Price Classification and Prediction

This GitHub repository hosts the Python notebooks and scripts accompanying the research paper "Stock Market Prediction using Machine Learning Techniques" published in the International Journal of Intelligent Systems and Applications in Engineering. The repository includes detailed exploratory data analysis (EDA), model building, model evaluation, and deployment code for predicting stock market trends and prices using various machine learning techniques. Below is a detailed explanation for each significant part of the code which can be used for your GitHub repository documentation.
Paper Link: https://ijisae.org/index.php/IJISAE/article/view/4845

#### Notebook 1: Stock Price Classification and Prediction

This notebook details the methodology used for classifying and predicting stock prices as outlined in the research paper.

1. **Exploratory Data Analysis:**
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load and visualize stock data
    stock_data = pd.read_csv('path_to_data.csv')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stock_data, x='Date', y='Close')
    plt.title('Stock Closing Prices Over Time')
    ```
    - Loads stock market data.
    - Visualizes trends in closing prices over time.

2. **Feature Engineering:**
    ```python
    stock_data['SMA'] = stock_data['Close'].rolling(window=15).mean()  # Short moving average
    stock_data['LMA'] = stock_data['Close'].rolling(window=60).mean()  # Long moving average
    stock_data['Momentum'] = stock_data['Close'] - stock_data['Close'].shift(10)
    ```
    - Calculates short and long-term moving averages and momentum as new features.

3. **Model Building and Evaluation:**
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X = stock_data[['SMA', 'LMA', 'Momentum']]
    y = stock_data['Trend']  # Binary target: 1 for up, 0 for down
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```
    - Splits the data into training and testing sets.
    - Trains a RandomForest model to classify stock price trends.

4. **Results Visualization:**
    ```python
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.arange(len(predictions)), y=predictions, hue=y_test)
    plt.title('Model Predictions vs. Actual Trends')
    ```
    - Predicts stock trends on test data and visualizes predictions against actual trends.

#### Notebook 2: Stock EDA

This notebook provides a comprehensive exploratory data analysis for the stock data.

1. **Data Visualization:**
    ```python
    plt.figure(figsize=(10, 6))
    sns.histplot(stock_data['Volume'], kde=True)
    plt.title('Distribution of Trading Volume')
    ```
    - Visualizes the distribution of trading volumes in the stock data.

2. **Correlation Analysis:**
    ```python
    correlation_matrix = stock_data.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix of Stock Features')
    ```
    - Calculates and visualizes the correlation matrix to identify relationships between different stock features.

#### Python Script: deployment.py

This script includes the deployment of the model using Streamlit, enabling users to predict stock prices interactively.

1. **Streamlit Deployment:**
    ```python
    import streamlit as st

    st.title('Stock Price Prediction')
    user_input = st.text_input("Enter Stock Symbol")
    # Prediction logic based on user input...
    ```
    - Uses Streamlit to create a web application for the stock price prediction model.
    - Allows users to enter stock symbols and get predictions directly through a web interface.

Each part of the repository is directly linked to the sections and methodologies discussed in the research paper, providing a practical implementation reference for readers and fellow researchers.
