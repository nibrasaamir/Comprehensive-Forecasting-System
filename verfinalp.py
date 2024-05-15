import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import load_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.svm import SVR
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import base64
from io import BytesIO
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
import dill as pickle

app = Flask(__name__)

# Load data
dfd = pd.read_csv("daily.csv")
dfd['date'] = pd.to_datetime(dfd['date'])
dfd.set_index('date', inplace=True)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Perform differencing
dfd_diff = dfd.diff().dropna()

# Train-test split
train_size = int(len(dfd_diff) * 0.8)
df_train, df_test = dfd_diff.iloc[:train_size], dfd_diff.iloc[train_size:]

# Normalize the data
train_data_scaled = scaler.transform(df_train)
test_data_scaled = scaler.transform(df_test)

# Load ARIMA model
with open('arima_model.pkl', 'rb') as f:
    arima_model_fit = pickle.load(f)

# Load SARIMA model
with open('sarima_model.pkl', 'rb') as f:
    sarima_model_fit = pickle.load(f)

# Load ETS model
with open('ets_model.pkl', 'rb') as f:
    ets_model_fit = pickle.load(f)



# Load Prophet model
with open('prophet_model.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

# Features and target for ANN model
X_train, X_test, y_train, y_test = train_test_split(df_train[['cycle', 'trend']], df_train['cycle'], test_size=0.2, random_state=42)

# Scale data for ANN model
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load ANN model
ann_model = load_model('ann_model.h5')

# Load SVR model
with open('svr_model.pkl', 'rb') as f:
    svr_model = pickle.load(f)

# Load LSTM model
lstm_model = load_model('lstm_model.h5', custom_objects={'mse': 'mse'})

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
from hybridmodel import HybridModel
# load hrbrid:
with open('hybrid_model_instance.pkl', 'rb') as f:
    hybridd_model = pickle.load(f)

#-----------------------------------------------------------------------------------------------------------------------------------------------



def plotArima(arima_forecast , df_test , df_train):
    # Calculate residuals
    residuals = df_test['cycle'] - arima_forecast

    # Plotting
    plt.figure(figsize=(12, 12))

    # Plot actual vs. forecast with enhanced style
    plt.subplot(3, 2, 1)
    plt.plot(df_test.index, df_test['cycle'], label='Actual', color='#007acc', linewidth=2)
    plt.plot(df_test.index, arima_forecast, label='ARIMA Forecast', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, arima_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.title('ARIMA - Actual vs Forecast', fontsize=14)
    plt.legend()

    # Plot train, test, and forecasted data with modern aesthetics
    plt.subplot(3, 2, 2)
    plt.plot(df_train.index, df_train['cycle'], label='Train Data', color='#009688', linewidth=2)
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', color='#007acc', linewidth=2)
    plt.plot(df_test.index, arima_forecast, label='Forecasted Data', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, arima_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.title('ARIMA Forecasting', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.legend()

    # Plot PACF of residuals
    plt.subplot(3, 2, 3)
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)

    # Bell-shaped histogram (Density plot) of residuals
    plt.subplot(3, 2, 4)
    sns.histplot(residuals, kde=True, color='#009688')
    plt.title('Histogram of Residuals', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # Heatmap of Autocorrelation
    lag_corr = pd.concat([residuals.shift(i) for i in range(11)], axis=1)
    lag_corr.columns = ['t'] + [f't-{i}' for i in range(1, 11)]
    plt.subplot(3, 2, 5)
    sns.heatmap(lag_corr.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Autocorrelation', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Lag', fontsize=12)

    # Scatter plot of actual vs. forecasted values
    plt.subplot(3, 2, 6)
    plt.scatter(df_test['cycle'], arima_forecast, color='#e63b2e', alpha=0.8)
    plt.plot(df_test['cycle'], df_test['cycle'], color='#007acc', linestyle='--', linewidth=2)
    plt.title('Actual vs Forecasted Values', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Forecasted Values', fontsize=12)

    # Customize layout and appearance
    plt.tight_layout()
    plt.style.use('seaborn-dark')  # Apply a modern seaborn style
    plt.grid(False)  # Disable gridlines
    plt.xticks(fontsize=10)  # Adjust tick label font size
    plt.yticks(fontsize=10)


    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast

def plotSarima(sarima_forecast , df_test , df_train):
    residuals = df_test['cycle'] - sarima_forecast
    plt.figure(figsize=(12, 12))

    # Plot actual vs. forecast with enhanced style
    plt.subplot(3, 2, 1)
    plt.plot(df_test.index, df_test['cycle'], label='Actual', color='#007acc', linewidth=2)
    plt.plot(df_test.index, sarima_forecast, label='SARIMA Forecast', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, sarima_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.title('SARIMA - Actual vs Forecast', fontsize=14)
    plt.legend()

    # Plot train, test, and forecasted data with modern aesthetics
    plt.subplot(3, 2, 2)
    plt.plot(df_train.index, df_train['cycle'], label='Train Data', color='#009688', linewidth=2)
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', color='#007acc', linewidth=2)
    plt.plot(df_test.index, sarima_forecast, label='Forecasted Data', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, sarima_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.title('SARIMA Forecasting', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.legend()

    # Plot PACF of residuals
    plt.subplot(3, 2, 3)
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)

    # Bell-shaped histogram (Density plot) of residuals
    plt.subplot(3, 2, 4)
    sns.histplot(residuals, kde=True, color='#009688')
    plt.title('Histogram of Residuals', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # Heatmap of Autocorrelation
    lag_corr = pd.concat([residuals.shift(i) for i in range(11)], axis=1)
    lag_corr.columns = ['t'] + [f't-{i}' for i in range(1, 11)]
    plt.subplot(3, 2, 5)
    sns.heatmap(lag_corr.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Autocorrelation', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Lag', fontsize=12)

    # Scatter plot of actual vs. forecasted values
    plt.subplot(3, 2, 6)
    plt.scatter(df_test['cycle'], sarima_forecast, color='#e63b2e', alpha=0.8)
    plt.plot(df_test['cycle'], df_test['cycle'], color='#007acc', linestyle='--', linewidth=2)
    plt.title('Actual vs Forecasted Values', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Forecasted Values', fontsize=12)

    # Customize layout and appearance
    plt.tight_layout()
    plt.style.use('seaborn-dark')  # Apply a modern seaborn style
    plt.grid(False)  # Disable gridlines
    plt.xticks(fontsize=10)  # Adjust tick label font size
    plt.yticks(fontsize=10)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast

       

def plotETS(ets_forecast , df_test , df_train):
    residuals = df_test['cycle'] - ets_forecast
    plt.figure(figsize=(12, 12))

    # Plot actual vs. forecast with enhanced style
    plt.subplot(3, 2, 1)
    plt.plot(df_test.index, df_test['cycle'], label='Actual', color='#007acc', linewidth=2)
    plt.plot(df_test.index, ets_forecast, label='ETS Forecast', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, ets_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.title('ETS - Actual vs Forecast', fontsize=14)
    plt.legend()

    # Plot train, test, and forecasted data with modern aesthetics
    plt.subplot(3, 2, 2)
    plt.plot(df_train.index, df_train['cycle'], label='Train Data', color='#009688', linewidth=2)
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', color='#007acc', linewidth=2)
    plt.plot(df_test.index, ets_forecast, label='Forecasted Data', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, ets_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.title('ETS Forecasting', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.legend()

    # Plot PACF of residuals
    plt.subplot(3, 2, 3)
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)

    # Bell-shaped histogram (Density plot) of residuals
    plt.subplot(3, 2, 4)
    sns.histplot(residuals, kde=True, color='#009688')
    plt.title('Histogram of Residuals', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # Heatmap of Autocorrelation
    lag_corr = pd.concat([residuals.shift(i) for i in range(11)], axis=1)
    lag_corr.columns = ['t'] + [f't-{i}' for i in range(1, 11)]
    plt.subplot(3, 2, 5)
    sns.heatmap(lag_corr.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Autocorrelation', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Lag', fontsize=12)

    # Scatter plot of actual vs. forecasted values
    plt.subplot(3, 2, 6)
    plt.scatter(df_test['cycle'], ets_forecast, color='#e63b2e', alpha=0.8)
    plt.plot(df_test['cycle'], df_test['cycle'], color='#007acc', linestyle='--', linewidth=2)
    plt.title('Actual vs Forecasted Values', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Forecasted Values', fontsize=12)

    # Customize layout and appearance
    plt.tight_layout()
    plt.style.use('seaborn-dark')  # Apply a modern seaborn style
    plt.grid(False)  # Disable gridlines
    plt.xticks(fontsize=10)  # Adjust tick label font size
    plt.yticks(fontsize=10)


    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast


def plotProphet(prophet_forecast , df_test , df_train):
    prophet_forecast = prophet_forecast['yhat'].tail(len(df_test))
   # prophet_forecast = pd.to_datetime(prophet_forecast)
   # residuals = df_test['cycle'] - prophet_forecast
    
    plt.figure(figsize=(12, 12))
#     prophet_forecast = pd.to_datetime(prophet_forecast)

#     residuals = df_test['cycle'] - prophet_forecast
#    # residuals = df_test['cycle'] - prophet_forecast
#     plt.figure(figsize=(12, 12))

    # Plot actual vs. forecast with enhanced style
    plt.subplot(3, 2, 1)
    plt.plot(df_test.index, df_test['cycle'], label='Actual', color='#007acc', linewidth=2)
    plt.plot(df_test.index, prophet_forecast, label='ETS Forecast', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, prophet_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.title('PROPHET - Actual vs Forecast', fontsize=14)
    plt.legend()
    # Plot actual vs. forecast with enhanced style
   

    # Plot train, test, and forecasted data with modern aesthetics
    plt.subplot(3, 2, 2)
    plt.plot(df_train.index, df_train['cycle'], label='Train Data', color='#009688', linewidth=2)
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', color='#007acc', linewidth=2)
    plt.plot(df_test.index, prophet_forecast, label='Forecasted Data', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, prophet_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.title('PROPHRT Forecasting', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.legend()

    # # Plot PACF of residuals
    # plt.subplot(3, 2, 3)
    # plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    # plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    # plt.xlabel('Lag', fontsize=12)
    # plt.ylabel('PACF', fontsize=12)
    # plt.legend()


    # Scatter plot of actual vs. forecasted values
    plt.subplot(3, 2, 4)
    plt.scatter(df_test['cycle'], prophet_forecast, color='#e63b2e', alpha=0.8)
    plt.plot(df_test['cycle'], df_test['cycle'], color='#007acc', linestyle='--', linewidth=2)
    plt.title('Actual vs Forecasted Values', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Forecasted Values', fontsize=12)
    plt.legend()

    # Customize layout and appearance
    plt.tight_layout()
    plt.style.use('seaborn-dark')  # Apply a modern seaborn style
    plt.grid(False)  # Disable gridlines
    plt.xticks(fontsize=10)  # Adjust tick label font size
    plt.yticks(fontsize=10)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast

def plotSvr(df_test , df_train ,y_test , X_train_scaled ,X_test_scaled , y_train , X_test ,svr_predictions):
    svr_predictions_series = pd.Series(svr_predictions.reshape(-1), index=df_test.index[:len(svr_predictions)])
    plt.figure(figsize=(12, 12))
    # Plot 1
    plt.subplot(3, 2, 1)
    plt.plot(df_test.index[:len(y_test)], y_test, label='Actual')
    plt.plot(df_test.index[:len(svr_predictions)], svr_predictions, label='svr Prediction')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('SVR - Actual vs Prediction')
    plt.legend()

    # Plot 2
    plt.subplot(3, 2, 2)
    plt.plot(df_train.index, df_train, label='Train Data')
    plt.plot(df_test.index, df_test, label='Test Data')
    plt.plot(svr_predictions_series.index, svr_predictions_series, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('SVR - Train, Test, and Forecasted Data')
    plt.legend()

    # Plot 4
    plt.subplot(3, 2, 3)
    mean_prediction = np.mean(svr_predictions)
    std_deviation = np.std(svr_predictions)
    upper_bound = mean_prediction + 1.96 * std_deviation
    lower_bound = mean_prediction - 1.96 * std_deviation
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', linestyle="--")
    plt.plot(df_test.index, mean_prediction * np.ones_like(df_test['cycle']), label='Mean Prediction', linestyle='-.')
    plt.fill_between(df_test.index, lower_bound * np.ones_like(df_test['cycle']), upper_bound * np.ones_like(df_test['cycle']), alpha=0.3, color='orange', label='95% Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('SVR - Mean Prediction and 95% Prediction Interval')
    plt.legend()


    plt.subplot(3, 2, 4)
    # Assuming 'residuals' is already calculated
    residuals = y_test - svr_predictions.reshape(-1)
    # Plot PACF of residuals
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)

    plt.subplot(3, 2, 5)
    # Bell-shaped histogram (Density plot) of residuals
    sns.histplot(residuals, kde=True, color='#009688')
    plt.title('Histogram of Residuals', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast


def plotLstm(df_test , df_train ,y_test , X_train_scaled ,X_test_scaled , y_train , X_test ,lstm_predictions):
    lstm_predictions_series = pd.Series(lstm_predictions.reshape(-1), index=df_test.index[:len(lstm_predictions)])
    plt.figure(figsize=(12, 12))
    # Plot 1
    plt.subplot(4, 2, 1)
    plt.plot(df_test.index[:len(y_test)], y_test, label='Actual')
    plt.plot(df_test.index[:len(lstm_predictions)], lstm_predictions, label='LSTM Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('LSTM - Actual vs Prediction')
    plt.legend()

    # Plot 2
    plt.subplot(4, 2, 2)
    plt.plot(df_train.index, df_train, label='Train Data')
    plt.plot(df_test.index, df_test, label='Test Data')
    plt.plot(lstm_predictions_series.index, lstm_predictions_series, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('LSTM - Train, Test, and Forecasted Data')
    plt.legend()

    # Plot 3
    plt.subplot(4, 2, 3)
    residuals = y_test - lstm_predictions.reshape(-1)
    plt.plot(df_test.index[:len(residuals)], residuals, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.title('Residuals Plot')
    plt.legend()

    # Plot 4
    plt.subplot(4, 2, 4)
    mean_prediction = np.mean(lstm_predictions)
    std_deviation = np.std(lstm_predictions)
    upper_bound = mean_prediction + 1.96 * std_deviation
    lower_bound = mean_prediction - 1.96 * std_deviation
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', linestyle="--")
    plt.plot(df_test.index, mean_prediction * np.ones_like(df_test['cycle']), label='Mean Prediction', linestyle='-.')
    plt.fill_between(df_test.index, lower_bound * np.ones_like(df_test['cycle']), upper_bound * np.ones_like(df_test['cycle']), alpha=0.3, color='orange', label='95% Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('LSTM - Mean Prediction and 95% Prediction Interval')
    plt.legend()


    plt.subplot(4, 2, 5)
    # Assuming 'residuals' is already calculated
    residuals = y_test - lstm_predictions.reshape(-1)
    # Plot PACF of residuals
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)

    plt.subplot(4, 2, 6)
    # Bell-shaped histogram (Density plot) of residuals
    sns.histplot(residuals, kde=True, color='#009688')
    plt.title('Histogram of Residuals', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast




def plotHybrid(df_test , hybrid_forecast , df_train):
    # Plot Hybrid Model predictions
    residuals = df_test['cycle'] - hybrid_forecast
    plt.figure(figsize=(12, 12))

    # Plot actual vs. forecast with enhanced style
    plt.subplot(3, 2, 1)
    plt.plot(df_test.index, df_test['cycle'], label='Actual', color='#007acc', linewidth=2)
    plt.plot(df_test.index, hybrid_forecast, label='HYBRID Forecast', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, hybrid_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.title('HYBRID - Actual vs Forecast', fontsize=14)
    plt.legend()

    # Plot train, test, and forecasted data with modern aesthetics
    plt.subplot(3, 2, 2)
    plt.plot(df_train.index, df_train['cycle'], label='Train Data', color='#009688', linewidth=2)
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', color='#007acc', linewidth=2)
    plt.plot(df_test.index, hybrid_forecast, label='Forecasted Data', color='#e63b2e', linestyle='--', linewidth=2)
    plt.fill_between(df_test.index, hybrid_forecast, df_test['cycle'], color='#e63b2e', alpha=0.1)
    plt.title('HYBRID Forecasting', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cycle Value', fontsize=12)
    plt.legend()

    # Plot PACF of residuals
    plt.subplot(3, 2, 3)
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)
    plt.legend()


    # Scatter plot of actual vs. forecasted values
    plt.subplot(3, 2, 4)
    plt.scatter(df_test['cycle'], hybrid_forecast, color='#e63b2e', alpha=0.8)
    plt.plot(df_test['cycle'], df_test['cycle'], color='#007acc', linestyle='--', linewidth=2)
    plt.title('Actual vs Forecasted Values', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Forecasted Values', fontsize=12)
    plt.legend()

    # Customize layout and appearance
    plt.tight_layout()
    plt.style.use('seaborn-dark')  # Apply a modern seaborn style
    plt.grid(False)  # Disable gridlines
    plt.xticks(fontsize=10)  # Adjust tick label font size
    plt.yticks(fontsize=10)


    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast






def plotAnn(df_test , df_train ,y_test , X_train_scaled ,X_test_scaled , y_train , X_test , ann_predictions):
    ann_predictions_series = pd.Series(ann_predictions.reshape(-1), index=df_test.index[:len(ann_predictions)])
    plt.figure(figsize=(12, 12))
    # Plot 1
    plt.subplot(4, 2, 1)
    plt.plot(df_test.index[:len(y_test)], y_test, label='Actual')
    plt.plot(df_test.index[:len(ann_predictions)], ann_predictions, label='ANN Prediction')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('ANN - Actual vs Prediction')
    plt.legend()

    # Plot 2
    plt.subplot(4, 2, 2)
    plt.plot(df_train.index, df_train, label='Train Data')
    plt.plot(df_test.index, df_test, label='Test Data')
    plt.plot(ann_predictions_series.index, ann_predictions_series, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('ANN - Train, Test, and Forecasted Data')
    plt.legend()

    # Plot 3
    plt.subplot(4, 2, 3)
    residuals = y_test - ann_predictions.reshape(-1)
    plt.plot(df_test.index[:len(residuals)], residuals, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.title('Residuals Plot')
    plt.legend()

    # Plot 4
    plt.subplot(4, 2, 4)
    mean_prediction = np.mean(ann_predictions)
    std_deviation = np.std(ann_predictions)
    upper_bound = mean_prediction + 1.96 * std_deviation
    lower_bound = mean_prediction - 1.96 * std_deviation
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', linestyle="--")
    plt.plot(df_test.index, mean_prediction * np.ones_like(df_test['cycle']), label='Mean Prediction', linestyle='-.')
    plt.fill_between(df_test.index, lower_bound * np.ones_like(df_test['cycle']), upper_bound * np.ones_like(df_test['cycle']), alpha=0.3, color='orange', label='95% Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('ANN - Mean Prediction and 95% Prediction Interval')
    plt.legend()

    # Plot 5
    plt.subplot(4, 2, 5)
    estimated_risk = mean_squared_error(y_test, ann_predictions)
    plt.plot(df_test.index, df_test['cycle'], label='Test Data', marker='o', linestyle='-')
    plt.plot(df_test.index, estimated_risk * np.ones_like(df_test['cycle']), label='Estimated Risk Prediction', marker='*', linestyle='-.')
    plt.xlabel('Date')
    plt.ylabel('Estimated Risk')
    plt.title('ANN - Estimated Risk Prediction')
    plt.legend()

   
    plt.subplot(4, 2, 6)
    # Assuming 'residuals' is already calculated
    residuals = y_test - ann_predictions.reshape(-1)
    # Plot PACF of residuals
    plot_pacf(residuals, lags=30, ax=plt.gca(), color='purple')
    plt.title('Partial Autocorrelation of Residuals', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('PACF', fontsize=12)

    plt.subplot(4, 2, 7)
    # Bell-shaped histogram (Density plot) of residuals
    sns.histplot(residuals, kde=True, color='#009688')
    plt.title('Histogram of Residuals', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'
    
    return plot_html_forecast

import pickle

def plot_all_models(df_train, df_test, train_data, test_data):
    rmse_results = {}

    # Load ARIMA model
    with open('arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    arima_forecast = arima_model.forecast(steps=len(test_data))
    arima_rmse = np.sqrt(mean_squared_error(df_test['cycle'], arima_forecast))
    rmse_results['ARIMA'] = arima_rmse

    # Load SARIMA model
    with open('sarima_model.pkl', 'rb') as f:
        sarima_model = pickle.load(f)
    sarima_forecast = sarima_model.forecast(steps=len(df_test))
    sarima_rmse = np.sqrt(mean_squared_error(df_test['cycle'], sarima_forecast))
    rmse_results['SARIMA'] = sarima_rmse

    # Load ETS model
    with open('ets_model.pkl', 'rb') as f:
        ets_model = pickle.load(f)
    ets_forecast = ets_model.forecast(steps=len(df_test))
    ets_rmse = np.sqrt(mean_squared_error(df_test['cycle'], ets_forecast))
    rmse_results['ETS'] = ets_rmse

    # Load Prophet model
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    future = prophet_model.make_future_dataframe(periods=len(df_test))
    prophet_forecast = prophet_model.predict(future)
    prophet_rmse = np.sqrt(mean_squared_error(df_test['cycle'], prophet_forecast['yhat'].tail(len(df_test))))
    rmse_results['Prophet'] = prophet_rmse

    # Load ANN model
    ann_model = load_model('ann_model.h5')
    ann_predictions = ann_model.predict(X_test_scaled)
    ann_rmse = np.sqrt(mean_squared_error(y_test, ann_predictions))
    rmse_results['ANN'] = ann_rmse

    # Load SVR model
    with open('svr_model.pkl', 'rb') as f:
        svr_model = pickle.load(f)
    svr_predictions = svr_model.predict(X_test_scaled)
    svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))
    rmse_results['SVR'] = svr_rmse

    # Load LSTM model
    lstm_model = load_model('lstm_model.h5', custom_objects={'mse': 'mse'})
    lstm_predictions = lstm_model.predict(X_test_lstm)
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    rmse_results['LSTM'] = lstm_rmse

    # Load Hybrid model
    with open('hybrid_model_instance.pkl', 'rb') as f:
        hybridd_model = pickle.load(f)
    hybrid_forecast = hybridd_model.forecast(df_train['cycle'], df_test['cycle'])    
    # hybrid_forecast = hybridd_model.forecast(steps=len(df_test))
    hybrid_rmse = np.sqrt(mean_squared_error(df_test['cycle'], hybrid_forecast))
    rmse_results['Hybrid'] = hybrid_rmse

    return rmse_results

def plot_date_range(df, start_date, end_date):
    subset = df.loc[start_date:end_date]
    
    # Plot trends
    subset['cycle'].plot(label='cycle')
    
    # Load ARIMA model
    with open('arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    arima_forecast = arima_model.predict(start=len(df_train), end=len(df_train) + len(subset) - 1)
    plt.plot(subset.index, arima_forecast, label='ARIMA Forecast')
    
    # Load SARIMA model
    with open('sarima_model.pkl', 'rb') as f:
        sarima_model = pickle.load(f)
    sarima_forecast = sarima_model.predict(start=len(df_train), end=len(df_train) + len(subset) - 1)
    plt.plot(subset.index, sarima_forecast, label='SARIMA Forecast')
    
    # Load ETS model
    with open('ets_model.pkl', 'rb') as f:
        ets_model = pickle.load(f)
    ets_forecast = ets_model.predict(start=len(df_train), end=len(df_train) + len(subset) - 1)
    plt.plot(subset.index, ets_forecast, label='ETS Forecast')
    
    # Load ANN model
    ann_model = load_model('ann_model.h5')
    X_subset = subset[['cycle', 'trend']]
    X_subset_scaled = scaler.transform(X_subset)
    ann_predictions = ann_model.predict(X_subset_scaled)
    plt.plot(subset.index, ann_predictions, label='ANN Prediction')
    
    # Load Prophet model
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    future = prophet_model.make_future_dataframe(periods=len(subset))
    prophet_forecast = prophet_model.predict(future)
    plt.plot(subset.index, prophet_forecast['yhat'].tail(len(subset)), label='Prophet Forecast')
    
    # Load SVR model
    with open('svr_model.pkl', 'rb') as f:
        svr_model = pickle.load(f)
    svr_predictions = svr_model.predict(X_subset_scaled)
    plt.plot(subset.index, svr_predictions, label='SVR Forecast')
    
    # Load LSTM model
    lstm_model = load_model('lstm_model.h5', custom_objects={'mse': 'mse'})
    lstm_predictions = lstm_model.predict(X_subset_scaled.reshape((X_subset_scaled.shape[0], 1, X_subset_scaled.shape[1])))
    plt.plot(subset.index, lstm_predictions, label='LSTM Forecast')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Trends and Predictions')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri_forecast = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html_forecast_dates = f'<img src="data:image/png;base64,{plot_data_uri_forecast}"/>'

    return plot_html_forecast_dates


#------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/api/daily', methods=['GET', 'POST'])
def get_daily_data():
    global dfd  # Assuming dfd is your data frame

    if request.method == 'POST':
        # Get form data from request
        see_acf_pacf = request.form.get('see_acf_pacf', 'no')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        model_choice = request.form.get('model_choice', '')
        see_best_model = request.form.get('see_best_model', 'no')
        html_forecast_dates = request.form.get('html_forecast_dates', 'no')
        see_all_rmse = request.form.get('see_all_rmse', 'no')
        see_csv_file = request.form.get('see_csv_file', 'no')
    else:
        html_forecast_dates = 'no'
        see_acf_pacf = 'no'
        start_date = None
        end_date = None
        model_choice = ''
        see_best_model = 'no'
        see_all_rmse = 'no'
        see_csv_file = 'no'

    if start_date is None:
        start_date = dfd.index.min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = dfd.index.max().strftime("%Y-%m-%d")

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    dfd_filtered = dfd.loc[start_date:end_date]
    dfd_filtered.dropna(inplace=True)

    plot_html_forecast_dates = ""

    # plot_html_forecast_dates = plot_date_range(dfd_filtered, start_date, end_date)
    if html_forecast_dates.lower() == 'yes':
        plot_html_forecast_dates = plot_date_range(dfd_filtered, start_date, end_date)
    
    plot_html_acf_pacf = ""
    plot_html_forecast = ""
    rmse_table_html = ""
    explanation_html = ""
    best_model_plot_label = ""
    model_accuracies = {}
    model_plots = {}

    arima_rmse = 0
    sarima_rmse = 0
    ets_rmse = 0
    prophet_rmse = 0
    ann_rmse = 0
    svr_rmse = 0
    lstm_rmse = 0
    hybrid_rmse = 0

    if see_acf_pacf.lower() == 'yes':
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(dfd_filtered['cycle'], lags=30, ax=axes[0])
        axes[0].set_title('ACF Plot - CO2 Data')
        plot_pacf(dfd_filtered['cycle'], lags=30, ax=axes[1])
        axes[1].set_title('PACF Plot - CO2 Data')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data_uri_acf_pacf = base64.b64encode(buffer.read()).decode('utf-8')
        plot_html_acf_pacf = f'<img src="data:image/png;base64,{plot_data_uri_acf_pacf}"/>'
        



    if model_choice.lower() in ["all" , "arima", "sarima", "ets", "prophet", "ann", "svr", "lstm" , "hybrid"]:
        if model_choice.lower() == "all":
            arima_forecast = arima_model_fit.forecast(steps=len(test_data_scaled))
            arima_mae = mean_absolute_error(df_test['cycle'], arima_forecast)
            arima_rmse = np.sqrt(mean_squared_error(df_test['cycle'], arima_forecast))
            model_accuracies["ARIMA"] = f"MAE: {arima_mae}, RMSE: {arima_rmse}"
            model_plots["ARIMA"] = plotArima(arima_forecast, df_test, df_train)
      
            sarima_forecast = sarima_model_fit.forecast(steps=len(df_test))
            sarima_mae = mean_absolute_error(df_test['cycle'], sarima_forecast)
            sarima_rmse = np.sqrt(mean_squared_error(df_test['cycle'], sarima_forecast))
            model_accuracies["SARIMA"] = f"MAE: {sarima_mae}, RMSE: {sarima_rmse}"
            model_plots["SARIMA"] = plotSarima(sarima_forecast, df_test, df_train)
    
            ets_forecast = ets_model_fit.forecast(steps=len(df_test))
            ets_mae = mean_absolute_error(df_test['cycle'], ets_forecast)
            ets_rmse = np.sqrt(mean_squared_error(df_test['cycle'], ets_forecast))
            model_accuracies["ETS"] = f"MAE: {ets_mae}, RMSE: {ets_rmse}"
            model_plots["ETS"] = plotETS(ets_forecast, df_test, df_train)
      
            future = prophet_model.make_future_dataframe(periods=len(df_test))
            prophet_forecast = prophet_model.predict(future)
            prophet_mae = mean_absolute_error(df_test['cycle'], prophet_forecast['yhat'].tail(len(df_test)))
            prophet_rmse = np.sqrt(mean_squared_error(df_test['cycle'], prophet_forecast['yhat'].tail(len(df_test))))
            print(prophet_rmse)
            model_accuracies["Prophet"] = f"MAE: {prophet_mae}, RMSE: {prophet_rmse}"
            model_plots["Prophet"] = plotProphet(prophet_forecast, df_test, df_train)
   
            ann_predictions = ann_model.predict(X_test_scaled)
            ann_mse = mean_squared_error(y_test, ann_predictions)
            ann_rmse = np.sqrt(ann_mse)
            print(ann_rmse)
            ann_mae = mean_absolute_error(y_test, ann_predictions)
            model_accuracies["ANN"] = f"MAE: {ann_mae}, MSE: {ann_mse}, RMSE: {ann_rmse}"
            model_plots["ANN"] = plotAnn(df_test, df_train, y_test, X_train_scaled, X_test_scaled, y_train, X_test, ann_predictions)
     
            svr_predictions = svr_model.predict(X_test_scaled)
            svr_mse = mean_squared_error(y_test, svr_predictions)
            svr_rmse = np.sqrt(svr_mse)
            print(svr_rmse)
            svr_mae = mean_absolute_error(y_test, svr_predictions)
            model_accuracies["SVR"] = f"MAE: {svr_mae}, MSE: {svr_mse}, RMSE: {svr_rmse}"
            model_plots["SVR"] = plotSvr(df_test, df_train, y_test, X_train_scaled, X_test_scaled, y_train, X_test, svr_predictions)
       
            lstm_predictions = lstm_model.predict(X_test_lstm)
            lstm_mae = mean_absolute_error(y_test, lstm_predictions)
            lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
            print(lstm_rmse)
            model_accuracies["LSTM"] = f"MAE: {lstm_mae}, RMSE: {lstm_rmse}"
            model_plots["LSTM"] = plotLstm(df_test, df_train, y_test, X_train_scaled, X_test_scaled, y_train, X_test, lstm_predictions)

            # hybrid_forecast = hybridd_model(df_train['cycle'], df_test['cycle'])
            hybrid_forecast = hybridd_model.forecast(df_train['cycle'], df_test['cycle'])
            hybrid_rmse = np.sqrt(mean_squared_error(df_test['cycle'], hybrid_forecast))
            model_accuracies["hybrid"] = f" RMSE: {hybrid_rmse}"
            model_plots["HYBRID"] = plotHybrid(df_test , hybrid_forecast , df_train)
            # rmse_results['Hybrid'] = hybrid_rmse

        elif model_choice.lower() == "arima":
            arima_forecast = arima_model_fit.forecast(steps=len(test_data_scaled))
            arima_mae = mean_absolute_error(df_test['cycle'], arima_forecast)
            arima_rmse = np.sqrt(mean_squared_error(df_test['cycle'], arima_forecast))
            model_accuracies["ARIMA"] = f"MAE: {arima_mae}, RMSE: {arima_rmse}"
            model_plots["ARIMA"] = plotArima(arima_forecast, df_test, df_train)
        elif model_choice.lower() == "sarima":
            sarima_forecast = sarima_model_fit.forecast(steps=len(df_test))
            sarima_mae = mean_absolute_error(df_test['cycle'], sarima_forecast)
            sarima_rmse = np.sqrt(mean_squared_error(df_test['cycle'], sarima_forecast))
            model_accuracies["SARIMA"] = f"MAE: {sarima_mae}, RMSE: {sarima_rmse}"
            model_plots["SARIMA"] = plotSarima(sarima_forecast, df_test, df_train)
        elif model_choice.lower() == "ets":
            ets_forecast = ets_model_fit.forecast(steps=len(df_test))
            ets_mae = mean_absolute_error(df_test['cycle'], ets_forecast)
            ets_rmse = np.sqrt(mean_squared_error(df_test['cycle'], ets_forecast))
            model_accuracies["ETS"] = f"MAE: {ets_mae}, RMSE: {ets_rmse}"
            model_plots["ETS"] = plotETS(ets_forecast, df_test, df_train)
        elif model_choice.lower() == "prophet":
            future = prophet_model.make_future_dataframe(periods=len(df_test))
            prophet_forecast = prophet_model.predict(future)
            prophet_mae = mean_absolute_error(df_test['cycle'], prophet_forecast['yhat'].tail(len(df_test)))
            prophet_rmse = np.sqrt(mean_squared_error(df_test['cycle'], prophet_forecast['yhat'].tail(len(df_test))))
            print(prophet_rmse)
            model_accuracies["Prophet"] = f"MAE: {prophet_mae}, RMSE: {prophet_rmse}"
            model_plots["Prophet"] = plotProphet(prophet_forecast, df_test, df_train)
        elif model_choice.lower() == "ann":
            ann_predictions = ann_model.predict(X_test_scaled)
            ann_mse = mean_squared_error(y_test, ann_predictions)
            ann_rmse = np.sqrt(ann_mse)
            print(ann_rmse)
            ann_mae = mean_absolute_error(y_test, ann_predictions)
            model_accuracies["ANN"] = f"MAE: {ann_mae}, MSE: {ann_mse}, RMSE: {ann_rmse}"
            model_plots["ANN"] = plotAnn(df_test, df_train, y_test, X_train_scaled, X_test_scaled, y_train, X_test, ann_predictions)
        elif model_choice.lower() == "svr":
            svr_predictions = svr_model.predict(X_test_scaled)
            svr_mse = mean_squared_error(y_test, svr_predictions)
            svr_rmse = np.sqrt(svr_mse)
            print(svr_rmse)
            svr_mae = mean_absolute_error(y_test, svr_predictions)
            model_accuracies["SVR"] = f"MAE: {svr_mae}, MSE: {svr_mse}, RMSE: {svr_rmse}"
            model_plots["SVR"] = plotSvr(df_test, df_train, y_test, X_train_scaled, X_test_scaled, y_train, X_test, svr_predictions)
        elif model_choice.lower() == "lstm":
            lstm_predictions = lstm_model.predict(X_test_lstm)
            lstm_mae = mean_absolute_error(y_test, lstm_predictions)
            lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
            print(lstm_rmse)
            model_accuracies["LSTM"] = f"MAE: {lstm_mae}, RMSE: {lstm_rmse}"
            model_plots["LSTM"] = plotLstm(df_test, df_train, y_test, X_train_scaled, X_test_scaled, y_train, X_test, lstm_predictions)
        elif model_choice.lower() == 'hybrid':
            hybrid_forecast = hybridd_model.forecast(df_train['cycle'], df_test['cycle'])
            hybrid_rmse = np.sqrt(mean_squared_error(df_test['cycle'], hybrid_forecast))
            rmse_results['Hybrid'] = hybrid_rmse    
       
    if see_all_rmse.lower() == 'yes':
        rmse_results = plot_all_models(df_train, df_test, train_data_scaled, test_data_scaled)

        # rmse_results = {
        #     "ARIMA": arima_rmse,
        #     "SARIMA": sarima_rmse,
        #     "ETS": ets_rmse,
        #     "Prophet": prophet_rmse,
        #     "ANN": ann_rmse,
        #     "SVR": svr_rmse,
        #     "LSTM": lstm_rmse
        #}
        # Filter out None values
        #valid_rmse_results = {key: value for key, value in rmse_results.items() if value is not None}

        # Sort the filtered results
        rmse_sorted = sorted(rmse_results.items(), key=lambda x: x[1])

       # rmse_sorted = sorted(rmse_results.items(), key=lambda x: x[1])
        rmse_table_html = "<h2>RMSE for each model:</h2><table><tr><th>Model</th><th>RMSE</th></tr>"
        for model, rmse in rmse_sorted:
            rmse_table_html += f"<tr><td>{model}</td><td>{rmse}</td></tr>"
        rmse_table_html += "</table>"

    if see_best_model.lower() == 'yes':
        best_model, best_rmse = rmse_sorted[0]  
        explanation_html = f"<p>{best_model} is the best model with RMSE: {best_rmse}</p>"
        best_model_plot_label = "Plot for the Best Model"

    return render_template('index1.html', plot_html_forecast_dates=plot_html_forecast_dates, model_accuracies=model_accuracies, model_plots=model_plots, plot_html_acf_pacf=plot_html_acf_pacf, plot_html_forecast=plot_html_forecast, rmse_table_html=rmse_table_html, best_model_plot_label=best_model_plot_label, explanation_html=explanation_html)

if __name__ == "__main__":
    app.run(port=5001, debug=True)