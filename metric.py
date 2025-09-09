import torch
from sklearn.metrics import f1_score

def accuracy(outputs, targets):
    pred = outputs.argmax(dim=1)
    return (pred == targets).float().mean().item()

def macro_f1(outputs, targets):
    pred = outputs.argmax(dim=1).detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    return f1_score(y, pred, average='macro', zero_division=0)
