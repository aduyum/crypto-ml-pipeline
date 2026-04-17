import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

class QuantSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3):
        super().__init__()
        # GRU (Gated Recurrent Unit) is highly effective for noisy financial time-series
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, num_features)
        out, _ = self.gru(x)
        
        # We only care about the output of the final time-step for our prediction
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out)

class PyTorchSequenceClassifier:
    """A scikit-learn style wrapper for our PyTorch Sequence Network."""
    def __init__(self, input_dim, seq_length=12, epochs=25, lr=0.001, batch_size=64):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantSequenceModel(input_dim).to(self.device)
        
        # TODO: Add class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4) # AdamW for better regularization

    def _create_sequences(self, X):
        """
        Converts 2D tabular data into 3D sequence data: (batch, seq_length, features).
        Pads the beginning so the output length matches the input length perfectly.
        """
        # Pad the first sequence_length-1 rows with the first row to maintain array length
        pad = np.repeat(X[0:1], self.seq_length - 1, axis=0)
        X_padded = np.vstack([pad, X])
        
        sequences =[]
        for i in range(len(X)):
            sequences.append(X_padded[i : i + self.seq_length])
        return np.array(sequences)

    def fit(self, X, y):
        X_seq = self._create_sequences(X)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                # Gradient clipping to prevent exploding gradients in RNNs
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

    def predict_proba(self, X):
        self.model.eval()
        X_seq = self._create_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)