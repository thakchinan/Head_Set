import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, criterion, metrics, optimizer,
                 config, data_loader, fold_id, valid_data_loader, class_weights):
        self.model = model
        self.criterion = lambda out, tgt: criterion(out, tgt, class_weights.to(out.device))
        self.metrics = metrics
        self.optimizer = optimizer
        self.cfg = config
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.fold_id = fold_id
        self.epochs = config['trainer']['epochs']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.sched = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)

    def _run_one_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, total_n = 0.0, 0
        all_out, all_tgt = [], []
        pbar = tqdm(loader, disable=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.criterion(out, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)
            all_out.append(out.detach().cpu())
            all_tgt.append(y.detach().cpu())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        all_out = torch.cat(all_out, 0)
        all_tgt = torch.cat(all_tgt, 0)
        logs = {"loss": total_loss / total_n}
        for m in self.metrics:
            logs[m.__name__] = float(m(all_out, all_tgt))
        return logs

    def train(self):
        best_score, best_state = -1, None
        for ep in range(1, self.epochs + 1):
            tr = self._run_one_epoch(self.train_loader, train=True)
            va = self._run_one_epoch(self.val_loader,   train=False)
            score = va.get("macro_f1", va.get("accuracy", 0.0))
            self.sched.step(score)
            print(f"[Fold {self.fold_id}] Epoch {ep}: TR {tr} | VA {va}")
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(self.model.state_dict(), f"best_fold{self.fold_id}.pt")
            print(f"Saved best model for fold {self.fold_id} with score {best_score:.4f}")
