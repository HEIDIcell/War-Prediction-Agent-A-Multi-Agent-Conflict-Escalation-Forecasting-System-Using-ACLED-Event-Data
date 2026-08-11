
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data.load_cases import FEATURE_COLUMNS

def make_sequences(df: pd.DataFrame, sequence_length: int = 6):
    X_list, y_list, idx_list = [], [], []
    for _, group in df.groupby("dyad"):
        group = group.sort_values("month_dt").reset_index()
        values = group[FEATURE_COLUMNS].values
        labels = group["label"].values
        original_indices = group["index"].values
        for i in range(sequence_length - 1, len(group)):
            X_list.append(values[i - sequence_length + 1:i + 1])
            y_list.append(labels[i])
            idx_list.append(original_indices[i])
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32), np.asarray(idx_list)

class LSTMProbabilityModel:
    def __init__(self, sequence_length=6, hidden_size=24, epochs=3, lr=0.01, seed=42):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("PyTorch is required for Baseline 2 LSTM. Install with `pip install torch`.") from exc

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_df = train_df.copy()
        test_df = test_df.copy()

        self.imputer.fit(train_df[FEATURE_COLUMNS])
        self.scaler.fit(self.imputer.transform(train_df[FEATURE_COLUMNS]))
        # Create float copies to avoid pandas dtype warnings when replacing integer columns.
        transformed_frames = []
        for frame in (train_df, test_df):
            frame = frame.copy()
            arr = self.imputer.transform(frame[FEATURE_COLUMNS])
            arr = self.scaler.transform(arr).astype(float)
            for j, col in enumerate(FEATURE_COLUMNS):
                frame[col] = arr[:, j]
            transformed_frames.append(frame)
        train_df, test_df = transformed_frames

        X_train, y_train, _ = make_sequences(train_df, self.sequence_length)
        X_test, _, test_original_indices = make_sequences(test_df, self.sequence_length)
        output = np.full(len(test_df), 0.5, dtype=float)
        if len(X_train) < 10 or len(X_test) == 0:
            return output

        pos_rate = float(y_train.mean())
        pos_weight_value = (1 - pos_rate) / max(pos_rate, 1e-3)

        class Net(nn.Module):
            def __init__(self, n_features, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
                self.dropout = nn.Dropout(0.15)
                self.out = nn.Linear(hidden_size, 1)
            def forward(self, x):
                _, (h, _) = self.lstm(x)
                h = self.dropout(h[-1])
                return self.out(h).squeeze(-1)

        model = Net(len(FEATURE_COLUMNS), self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32))
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy()

        test_index_to_position = {idx: pos for pos, idx in enumerate(test_df.index.values)}
        for original_idx, p in zip(test_original_indices, probs):
            if original_idx in test_index_to_position:
                output[test_index_to_position[original_idx]] = float(p)
        return np.clip(output, 0.001, 0.999)
