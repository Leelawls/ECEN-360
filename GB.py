import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb
import random


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def GB(df):
    set_seed(77)

    # ---- 1. Setup ----
    target_col = df.columns[0]
    feature_cols = df.columns[1:].tolist()

    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Add time features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    feature_cols += ["hour", "dayofweek", "month", "dayofyear", "is_weekend"]

    df = df[[target_col] + feature_cols].copy()
    df = df.fillna(method="ffill").dropna()

    # ---- 2. Scale ----
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_data = scaler_x.fit_transform(df[feature_cols])
    y_data = scaler_y.fit_transform(df[[target_col]])

    # ---- 3. Sequence Maker (Multi-step) ----
    def create_sequences(X, y, window_size=48, forecast_horizon=6):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size - forecast_horizon + 1):
            X_seq.append(X[i : i + window_size].flatten())
            y_seq.append(
                y[i + window_size : i + window_size + forecast_horizon].flatten()
            )
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(X_data, y_data, window_size=48, forecast_horizon=6)

    # ---- 4. Split ----
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---- 5. Train XGBoost Multi-Output Model ----
    models = []
    y_preds_real = []
    y_tests_real = []

    for i in range(y_train.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.1,
            early_stopping_rounds=20,
            tree_method="hist",
            random_state=77,
            objective="reg:squarederror",
        )

        model.fit(
            X_train, y_train[:, i], eval_set=[(X_test, y_test[:, i])], verbose=False
        )

        y_pred = model.predict(X_test).reshape(-1, 1)
        y_pred_real = scaler_y.inverse_transform(y_pred)
        y_test_real = scaler_y.inverse_transform(y_test[:, i].reshape(-1, 1))

        models.append(model)
        y_preds_real.append(y_pred_real)
        y_tests_real.append(y_test_real)

    # ---- 6. Aggregate Predictions ----
    y_preds_real = np.hstack(y_preds_real)
    y_tests_real = np.hstack(y_tests_real)

    # ---- 7. Metrics ----
    mse = mean_squared_error(y_tests_real, y_preds_real)
    mae = mean_absolute_error(y_tests_real, y_preds_real)
    r2 = r2_score(y_tests_real, y_preds_real)
    r, _ = pearsonr(y_tests_real.ravel(), y_preds_real.ravel())

    print("\nMulti-Step Forecast Evaluation Metrics")
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"R   : {r:.4f}")

    # ---- 8. Plot (last horizon forecast only) ----
    # ---- 8. Plot (custom date range: 09-03 to 09-11, 2023) ----
    plot_start = pd.to_datetime("2023-09-03")
    plot_end = pd.to_datetime("2023-09-11")

    plot_index = df.index[-len(y_tests_real) :]
    y_test_plot_df = pd.DataFrame(
        y_tests_real[:, -1], index=plot_index, columns=["Actual"]
    )
    y_pred_plot_df = pd.DataFrame(
        y_preds_real[:, -1], index=plot_index, columns=["Predicted"]
    )

    plot_df = pd.concat([y_test_plot_df, y_pred_plot_df], axis=1)
    plot_df = plot_df.loc[plot_start:plot_end]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df.index, plot_df["Actual"], label="Actual", linewidth=2)
    plt.plot(
        plot_df.index,
        plot_df["Predicted"],
        label="Predicted (t+6)",
        linestyle="--",
        linewidth=2,
    )
    plt.title("XGBoost Forecast from 2023-09-03 to 2023-09-11", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()


def forecast_signal(csv_path, label):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df.index = df.index.tz_localize(None)
    df = df[[df.columns[0]]]
    df.columns = [label]

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    target_col = label
    feature_cols = ["hour", "dayofweek", "month", "dayofyear", "is_weekend"]

    df = df[[target_col] + feature_cols].copy()
    df = df.fillna(method="ffill").dropna()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_data = scaler_x.fit_transform(df[feature_cols])
    y_data = scaler_y.fit_transform(df[[target_col]])

    def create_sequences(X, y, window_size=48, forecast_horizon=6):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size - forecast_horizon + 1):
            X_seq.append(X[i : i + window_size].flatten())
            y_seq.append(
                y[i + window_size : i + window_size + forecast_horizon].flatten()
            )
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(X_data, y_data)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = []
    y_preds_real = []
    y_tests_real = []

    for i in range(y_train.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.1,
            early_stopping_rounds=20,
            tree_method="hist",
            random_state=77,
            objective="reg:squarederror",
        )

        model.fit(
            X_train, y_train[:, i], eval_set=[(X_test, y_test[:, i])], verbose=False
        )

        y_pred = model.predict(X_test).reshape(-1, 1)
        y_pred_real = scaler_y.inverse_transform(y_pred)
        y_test_real = scaler_y.inverse_transform(y_test[:, i].reshape(-1, 1))

        models.append(model)
        y_preds_real.append(y_pred_real)
        y_tests_real.append(y_test_real)

    y_preds_real = np.hstack(y_preds_real)
    y_tests_real = np.hstack(y_tests_real)
    index = df.index[-len(y_tests_real) :]

    return pd.DataFrame(
        {f"{label}_actual": y_tests_real[:, -1], f"{label}_pred": y_preds_real[:, -1]},
        index=index,
    )


def forecast_signal(csv_path, label):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df.index = df.index.tz_localize(None)
    df = df[[df.columns[0]]]
    df.columns = [label]

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    target_col = label
    feature_cols = ["hour", "dayofweek", "month", "dayofyear", "is_weekend"]

    df = df[[target_col] + feature_cols].copy()
    df = df.fillna(method="ffill").dropna()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_data = scaler_x.fit_transform(df[feature_cols])
    y_data = scaler_y.fit_transform(df[[target_col]])

    def create_sequences(X, y, window_size=48, forecast_horizon=6):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size - forecast_horizon + 1):
            X_seq.append(X[i : i + window_size].flatten())
            y_seq.append(
                y[i + window_size : i + window_size + forecast_horizon].flatten()
            )
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(X_data, y_data)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = []
    y_preds_real = []
    y_tests_real = []

    for i in range(y_train.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.1,
            early_stopping_rounds=20,
            tree_method="hist",
            random_state=77,
            objective="reg:squarederror",
        )

        model.fit(
            X_train, y_train[:, i], eval_set=[(X_test, y_test[:, i])], verbose=False
        )

        y_pred = model.predict(X_test).reshape(-1, 1)
        y_pred_real = scaler_y.inverse_transform(y_pred)
        y_test_real = scaler_y.inverse_transform(y_test[:, i].reshape(-1, 1))

        models.append(model)
        y_preds_real.append(y_pred_real)
        y_tests_real.append(y_test_real)

    y_preds_real = np.hstack(y_preds_real)
    y_tests_real = np.hstack(y_tests_real)
    index = df.index[-len(y_tests_real) :]

    return pd.DataFrame(
        {f"{label}_actual": y_tests_real[:, -1], f"{label}_pred": y_preds_real[:, -1]},
        index=index,
    )


def GB_Forecast_Netload():
    set_seed(77)
    load_df = forecast_signal("Load.csv", "Load")
    solar_df = forecast_signal("Solar.csv", "Solar")
    wind_df = forecast_signal("Wind.csv", "Wind")

    full_df = pd.concat([load_df, solar_df, wind_df], axis=1, join="inner")
    full_df["NetLoad_actual"] = (
        full_df["Load_actual"] - full_df["Solar_actual"] - full_df["Wind_actual"]
    )
    full_df["NetLoad_pred"] = (
        full_df["Load_pred"] - full_df["Solar_pred"] - full_df["Wind_pred"]
    )

    # Filter date range
    plot_df = full_df.loc["2023-09-10":"2023-09-11"]

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(
        plot_df.index, plot_df["NetLoad_actual"], label="Actual NetLoad", linewidth=2
    )
    plt.plot(
        plot_df.index,
        plot_df["NetLoad_pred"],
        label="Predicted NetLoad (t+6)",
        linestyle="--",
        linewidth=2,
    )
    plt.title("XGBoost NetLoad Forecast (2023-09-10 to 2023-09-12)", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Net Load (MW)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()


# Run forecast
GB_Forecast_Netload()
