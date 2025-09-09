import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

class NumpyEEGDataset(Dataset):
    """
    คาดว่าไฟล์ .npy 2 ไฟล์:
    - X.npy: shape [N, C, T]  เช่น [num_nights, 1, total_samples]
    - y.npy: shape [N]        labels: 0=poor,1=fair,2=good
    แนะนำให้ preprocess (band-pass, z-score) มาก่อน
    """
    def __init__(self, X, y, dtype=torch.float32):
        self.X = X
        self.y = y
        self.dtype = dtype

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def load_folds_data(np_dir, num_folds=5):
    X = np.load(f"{np_dir}/X.npy")  # [N,C,T]
    y = np.load(f"{np_dir}/y.npy")  # [N]
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = []
    for tr_idx, va_idx in skf.split(X, y):
        folds.append((tr_idx, va_idx))
    return (X, y), folds

def load_folds_data_shhs(np_dir, num_folds=5):
    # กรณีโครงสร้างชื่อไฟล์ต่าง ก็แก้ตรงนี้
    return load_folds_data(np_dir, num_folds)

def data_generator_np(fold_indices_train, fold_indices_val, batch_size, X=None, y=None):
    # หมายเหตุ: main.py โหลด folds_data global แล้วส่งเข้ามาเป็น indices
    # เพื่อให้ใช้ได้ ให้ฟังก์ชันนี้รับ X,y ผ่าน closure/global หรือปรับ signature
    global GLOBAL_X, GLOBAL_Y
    tr_ds = NumpyEEGDataset(GLOBAL_X[fold_indices_train], GLOBAL_Y[fold_indices_train])
    va_ds = NumpyEEGDataset(GLOBAL_X[fold_indices_val],   GLOBAL_Y[fold_indices_val])

    data_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # นับจำนวนตัวอย่างต่อคลาสสำหรับ class weight
    ids = GLOBAL_Y[fold_indices_train]
    data_count = np.bincount(ids, minlength=3)
    return data_loader, valid_loader, data_count
