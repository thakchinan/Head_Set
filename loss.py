import torch
import torch.nn as nn

def cross_entropy(outputs, targets, class_weights=None):
    # outputs: [B,3] logits, targets: [B] long
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn(outputs, targets)
