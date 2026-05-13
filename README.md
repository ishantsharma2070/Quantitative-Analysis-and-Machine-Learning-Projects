# Machine Learning, Data Science and Quantitative Analysis Projects

This repository contains a collection of Python-based projects focused on machine learning, natural language processing, time-series forecasting, quantitative finance, and predictive modelling.

The projects demonstrate practical experience with data preprocessing, feature engineering, statistical analysis, model training, evaluation, and visualisation.

## Projects Included

### 1. Statistical Modelling and Time-Series Analysis
- Ordinary Least Squares regression
- Augmented Dickey-Fuller stationarity testing
- Time-series diagnostics
- Financial data analysis using Python

### 2. Cryptocurrency Price Prediction Using LSTM
- Collected BTC/USDT market data using Binance API via `ccxt`
- Preprocessed closing price data using Min-Max scaling
- Built LSTM neural network models using TensorFlow and Keras
- Used train-validation-test split for model evaluation
- Evaluated predictions using RMSE and MAPE

### 3. Portfolio Optimisation
- Analysed historical stock data for AAPL, MSFT, GOOGL and AMZN
- Calculated expected returns, volatility and covariance matrix
- Implemented Markowitz Efficient Frontier
- Optimised portfolio weights using SciPy
- Visualised risk-return trade-offs and Sharpe ratio performance

### 4. SMS Spam Detection Using NLP
- Built a text classification model for spam detection
- Applied text preprocessing including tokenisation, stop-word removal and lemmatisation
- Used TF-IDF vectorisation
- Trained Random Forest and Gradient Boosting classifiers
- Evaluated model performance using accuracy, precision and recall

### 5. Sales Demand Prediction
- Developed a demand prediction model using structured sales data
- Performed exploratory data analysis on city, brand, capacity, price and seasonal trends
- Applied feature engineering including location clustering and cyclic time features
- Used XGBoost for predictive modelling
- Analysed the impact of price, population, seasonality and location on demand

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- XGBoost
- NLTK
- SciPy
- Statsmodels
- Matplotlib
- Seaborn
- yfinance
- ccxt

## Key Skills Demonstrated

- Machine learning
- Deep learning
- Natural language processing
- Time-series analysis
- Financial modelling
- Portfolio optimisation
- Data preprocessing
- Feature engineering
- Model evaluation
- Data visualisation
- API-based data collection

## Repository Structure

```text
.
├── Project 1.ipynb                         # Statistical modelling and time-series analysis
├── Project 2.ipynb                         # Cryptocurrency price prediction using LSTM
├── Project 3.ipynb                         # Portfolio optimisation
├── NLP.ipynb                               # SMS spam detection using NLP
├── Project 5 Predicting Sales demand.ipynb # Sales demand prediction
└── README.md
