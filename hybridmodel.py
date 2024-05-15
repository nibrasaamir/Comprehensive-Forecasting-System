# hybrid_model.py
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

class HybridModel:
    def __init__(self):
        self.arima_model = None
        self.ann_model = None
        self.scaler_ann = StandardScaler()

    def fit_arima(self, train_data):
        self.arima_model = ARIMA(train_data, order=(2, 1, 0)).fit()

    def fit_ann(self, train_data):
        X_train_ann = train_data.shift(1).dropna().values.reshape(-1, 1)
        y_train_ann = train_data.diff().shift(-1).dropna().values.reshape(-1, 1)
        X_train_ann_scaled = self.scaler_ann.fit_transform(X_train_ann)
        y_train_ann_scaled = self.scaler_ann.transform(y_train_ann)

        self.ann_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_ann_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.ann_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        self.ann_model.fit(X_train_ann_scaled, y_train_ann_scaled, epochs=300, batch_size=128, verbose=0)

    def forecast(self, train_data, test_data):
        arima_forecast = self.arima_model.forecast(len(test_data))
        arima_residuals = test_data - arima_forecast

        X_test_ann = test_data.shift(1).dropna().values.reshape(-1, 1)
        X_test_ann_scaled = self.scaler_ann.transform(X_test_ann)
        ann_residuals_scaled = self.ann_model.predict(X_test_ann_scaled)
        ann_residuals = self.scaler_ann.inverse_transform(ann_residuals_scaled).flatten()

        if len(ann_residuals) < len(test_data):
            ann_residuals = np.append(ann_residuals, [ann_residuals[-1]])  # Append last value to match length

        hybrid_forecast = arima_forecast + ann_residuals
        return hybrid_forecast



