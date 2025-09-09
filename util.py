import numpy as np
import torch

def calc_class_weight(data_count):
    # data_count: [3] นับตัวอย่างของแต่ละคลาสใน train fold
    total = data_count.sum()
    weights = total / (3 * np.maximum(data_count, 1))
    return torch.tensor(weights, dtype=torch.float32)
