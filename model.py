import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG1DCNN_Head(nn.Module):
    def __init__(self, in_ch=1, n_classes=3):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,7, padding=3), nn.BatchNorm1d(128),nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,256,7,padding=3), nn.BatchNorm1d(256),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):  # x: [B, C, T]
        z = self.fe(x).squeeze(-1)  # [B,256]
        logits = self.head(z)       # [B,3]
        return logits
