import torch
import torch.nn as nn
import torch.nn.functional as F

# แบ่งเป็น windows ยาว win_len (ตัวอย่าง: 30 วินาที * 128 Hz = 3840 จุด)
def frame_signal(x, win_len, hop_len):
    # x: [B, C, T]
    B, C, T = x.shape
    n = 1 + (T - win_len) // hop_len
    frames = []
    for i in range(n):
        s = i * hop_len
        frames.append(x[:, :, s:s+win_len])
    return torch.stack(frames, dim=2)  # [B, C, N, win_len]

class EEGEncoder1D(nn.Module):
    def __init__(self, in_ch=1, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,7, padding=3), nn.BatchNorm1d(128),nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,256,7,padding=3), nn.BatchNorm1d(256),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(256, feat_dim)
    def forward(self, x):   # x: [B, C, L]
        f = self.net(x).squeeze(-1)  # [B, 256]
        return self.proj(f)          # [B, feat_dim]

class SleepQuality3C(nn.Module):
    def __init__(self, in_ch=1, feat_dim=256, num_classes=3,
                 win_len=3840, hop_len=3840):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.enc = EEGEncoder1D(in_ch=in_ch, feat_dim=feat_dim)
        self.temporal = nn.LSTM(input_size=feat_dim, hidden_size=128,
                                num_layers=1, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: [B, C, T]
        with torch.no_grad():
            frames = frame_signal(x, self.win_len, self.hop_len)  # [B, C, N, L]
        B, C, N, L = frames.shape
        frames = frames.permute(0, 2, 1, 3).contiguous().view(B*N, C, L)  # [B*N, C, L]
        feats = self.enc(frames)              # [B*N, feat_dim]
        feats = feats.view(B, N, -1)          # [B, N, feat_dim]
        seq_out, _ = self.temporal(feats)     # [B, N, 256]
        pooled = seq_out.mean(dim=1)          # temporal average pooling
        logits = self.head(pooled)            # [B, 3]
        return logits

# ====== ใช้งาน ======
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SleepQuality3C(in_ch=1, feat_dim=256, num_classes=3,
                       win_len=3840, hop_len=3840).to(device)

# class weights (ตัวอย่าง): poor:fair:good = 1:2:3 สลับตามสัดส่วนจริง
class_weights = torch.tensor([1.5, 1.0, 1.2], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ในลูปเทรน:
# logits = model(x)            # x: [B, C, T]
# loss = criterion(logits, y)  # y: [B] (0=Poor,1=Fair,2=Good)
# pred = logits.argmax(-1)
