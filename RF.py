import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def RF(df):
    set_seed(77)

    # ---- 1. Setup ----
    target_col = df.columns[0]
    feature_cols = df.columns[1:].tolist()

    # Make sure df index is datetime and sorted
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df = df[[target_col] + feature_cols].copy()
    df = df.fillna(method="ffill").dropna()

    # ---- 2. Scale ----
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # ---- 3. Sequence Maker ----
    def create_sequences(data, target_index=0, window_size=24):
        X, y = [], []
        feature_indices = list(range(data.shape[1]))
        feature_indices.remove(target_index)
        for i in range(len(data) - window_size):
            seq = data[i : i + window_size, feature_indices].flatten()
            target = data[i + window_size, target_index]
            X.append(seq)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)

    X, y = create_sequences(scaled_data, target_index=0, window_size=24)

    # ---- 4. Split ----
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---- 5. Train RF Model ----
    model = RandomForestRegressor(n_estimators=100, random_state=77)
    model.fit(X_train, y_train.ravel())

    # ---- 6. Predict and Inverse Scale ----
    y_pred_scaled = model.predict(X_test)
    dummy = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
    dummy[:, 0] = y_pred_scaled
    y_pred_real = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_test.ravel()
    y_test_real = scaler.inverse_transform(dummy)[:, 0]

    # ---- 7. Metrics ----
    mse = mean_squared_error(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)
    r, _ = pearsonr(y_test_real, y_pred_real)

    print("\nForecast Evaluation Metrics")
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"R   : {r:.4f}")

    # ---- 8. Plot (last month only, pretty) ----
    last_n = 24 * 30  # 30 days of hourly data
    plot_index = df.index[-len(y_test_real) :][-last_n:]
    y_test_real_plot = y_test_real[-last_n:]
    y_pred_real_plot = y_pred_real[-last_n:]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_index, y_test_real_plot, label="Actual", linewidth=2)
    plt.plot(
        plot_index, y_pred_real_plot, label="Predicted", linestyle="--", linewidth=2
    )
    plt.title("Forecast (Random Forest, Last Month - Hourly)", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()


RF(pd.read_csv("Wind.csv", index_col=0))
