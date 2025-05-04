import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Transformer(df):
    set_seed(77)

    # ---- 1. Setup ----
    target_col = df.columns[0]
    feature_cols = df.columns[1:].tolist()

    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    feature_cols += ["hour", "dayofweek", "is_weekend"]

    df = df[[target_col] + feature_cols].copy()
    df = df.fillna(method="ffill").dropna()

    # ---- 2. Scale ----
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_data = scaler_x.fit_transform(df[feature_cols])
    y_data = scaler_y.fit_transform(df[[target_col]])

    # ---- 3. Sequence Maker ----
    def create_sequences(X, y, window_size=48):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i : i + window_size])
            y_seq.append(y[i + window_size])
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(X_data, y_data, window_size=48)

    # ---- 4. Split ----
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---- 5. Convert to PyTorch ----
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # ---- 6. Transformer Model ----
    class TransformerForecast(nn.Module):
        def __init__(
            self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128
        ):
            super(TransformerForecast, self).__init__()
            self.embedding = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            # x: (batch_size, seq_len, input_size)
            x = self.embedding(x)  # (batch_size, seq_len, d_model)
            x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
            x = self.transformer(x)
            x = x[-1, :, :]  # Take last time step
            return self.fc(x)

    model = TransformerForecast(input_size=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---- 7. Train ----
    epochs = 15
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_test)
            val_loss = criterion(val_output, y_test)
            val_losses.append(val_loss.item())

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
        )

    # ---- 8. Predict and Inverse Scale ----
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
        y_test_scaled = y_test.numpy()

        y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
        y_test_real = scaler_y.inverse_transform(y_test_scaled)

    # ---- 9. Metrics ----
    mse = mean_squared_error(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)
    r, _ = pearsonr(y_test_real.ravel(), y_pred_real.ravel())

    print("\nForecast Evaluation Metrics")
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"R   : {r:.4f}")

    # ---- 10. Plot (last month only, pretty) ----
    last_n = 24 * 30  # 30 days of hourly data
    plot_index = df.index[-len(y_test_real) :][-last_n:]
    y_test_real_plot = y_test_real[-last_n:]
    y_pred_real_plot = y_pred_real[-last_n:]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_index, y_test_real_plot, label="Actual", linewidth=2)
    plt.plot(
        plot_index, y_pred_real_plot, label="Predicted", linestyle="--", linewidth=2
    )
    plt.title("Transformer Forecast (Last Month - Hourly)", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()
