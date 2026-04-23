# src/models_nn.py
import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class WideDeepNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.wide = nn.Linear(input_dim, 16)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.combined(torch.cat([self.wide(x), self.deep(x)], dim=1))

class AttentionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, 
                                               dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x, _ = self.attention(x, x, x)
        return self.fc(self.norm(x.squeeze(1)))

NEURAL_MODELS = {
    'SimpleDNN': SimpleDNN,
    'WideDeepNet': WideDeepNet,
    'AttentionNet': AttentionNet
}