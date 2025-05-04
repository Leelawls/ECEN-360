import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


class RNNTrainer:
    def __init__(self, model, lr=0.01, batch_size=32, max_grad_norm=1.0, device=None):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_loader(self, X, y):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, verbose=True):
        train_loader = self._create_loader(X_train, y_train)
        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            self.model.train()
            batch_losses = []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb).squeeze()
                loss = self.criterion(pred, yb.squeeze())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                batch_losses.append(loss.item())

            avg_train_loss = np.mean(batch_losses)
            self.train_losses.append(avg_train_loss)

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}"

            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                if verbose:
                    msg += f" | Val Loss: {val_loss:.4f}"

            if verbose:
                print(msg)

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.float32).to(self.device)

            pred = self.model(X).squeeze()
            loss = self.criterion(pred, y.squeeze())
        return loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X)
            return preds.cpu().numpy()

    def plot_forecast(self, X_test, y_test, y_scaler, datetime_index, n_plot=200):
        """
        X_test: test input sequences
        y_test: true y values (scaled)
        y_scaler: fitted scaler for inverse_transform
        datetime_index: datetime index aligned with y_test
        n_plot: number of samples to plot
        """
        preds = self.predict(X_test)
        preds_real = y_scaler.inverse_transform(preds)
        y_real = y_scaler.inverse_transform(y_test)

        # Plotting
        plt.figure(figsize=(14, 6))
        plt.plot(datetime_index[:n_plot], y_real[:n_plot], label="Actual", linewidth=2)
        plt.plot(
            datetime_index[:n_plot],
            preds_real[:n_plot],
            label="Predicted",
            linestyle="--",
        )
        plt.title("Forecast vs Actual")
        plt.xlabel("Datetime")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
