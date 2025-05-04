import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def SA(df):
    set_seed(77)

    # ---- 1. Setup ----
    target_col = df.columns[0]

    # Make sure df index is datetime and sorted
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[[target_col]].copy()
    df = df.fillna(method="ffill").dropna()

    y = df[target_col].values
    y_index = df.index

    # ---- 2. Split ----
    split = int(0.8 * len(y))
    y_train, y_test = y[:split], y[split:]
    index_train, index_test = y_index[:split], y_index[split:]

    # ---- 3. Train SARIMA Model ----
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    # ---- 4. Forecast ----
    y_pred = results.forecast(steps=len(y_test))

    # ---- 5. Metrics ----
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r, _ = pearsonr(y_test, y_pred)

    print("\nForecast Evaluation Metrics")
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"R   : {r:.4f}")

    # ---- 6. Plot (last month only, pretty) ----
    last_n = 24 * 30  # 30 days of hourly data
    y_test_plot = y_test[-last_n:]
    y_pred_plot = y_pred[-last_n:]
    plot_index = index_test[-last_n:]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_index, y_test_plot, label="Actual", linewidth=2)
    plt.plot(plot_index, y_pred_plot, label="Predicted", linestyle="--", linewidth=2)
    plt.title("SARIMA Forecast (Last Month - Hourly)", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()
