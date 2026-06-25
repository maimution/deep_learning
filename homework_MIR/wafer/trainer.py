"""Shared training engine for the heterogeneous backbone trio.

The three per-backbone scripts (train_effb1.py / train_convnext.py /
train_swin.py) each define only their own config and call run() here, so they
share one identical, well-tested pipeline:
  weighted sampler + effective-number focal loss + EMA + OneCycle + bf16.

Every backbone logs the same metrics:
  per epoch  -> val accuracy / val macro-F1
  final test -> accuracy / macro-F1 / MCC + per-class F1 (+ confusion matrix)
"""
import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

from dataset import WaferDataset
from model import build_model, FocalLoss

OUT = "data/processed"


class EMA:
    """Exponential moving average of weights -> steadier eval accuracy."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def make_loaders(size, bs, workers, seed, data_dir=OUT):
    path = f"{data_dir}/labeled_X_{size}.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run: python wafer/prepare_data.py --size {size}")
    X = np.load(path, mmap_mode="r")            # mmap: only the needed rows hit RAM
    y = np.load(f"{data_dir}/labeled_y.npy")
    sp = np.load(f"{data_dir}/split.npz")
    tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]

    ytr = y[tr]
    cls_count = np.bincount(ytr, minlength=9)
    # sqrt-inverse-frequency sampling: expose minority classes without starving 'none'
    sample_w = (1.0 / np.sqrt(cls_count + 1))[ytr]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(),
                                    num_samples=len(ytr), replacement=True)

    dtr = WaferDataset(np.ascontiguousarray(X[tr]), ytr, train=True, seed=seed)
    dte = WaferDataset(np.ascontiguousarray(X[te]), y[te], train=False)
    ltr = DataLoader(dtr, batch_size=bs, sampler=sampler, num_workers=workers,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    lte = DataLoader(dte, batch_size=256, num_workers=12, pin_memory=True)

    # Monitor loader for choosing the "best" checkpoint:
    #  - stratified split: a real validation set.
    #  - official split (no val): a fixed sub-sample of the TRAIN set (un-augmented),
    #    since we must not peek at the test set for model selection.
    if len(va) > 0:
        dmon = WaferDataset(np.ascontiguousarray(X[va]), y[va], train=False)
        mon_name = "val"
    else:
        rng = np.random.default_rng(seed)
        sub = tr if len(tr) <= 20000 else rng.choice(tr, 20000, replace=False)
        dmon = WaferDataset(np.ascontiguousarray(X[sub]), y[sub], train=False)
        mon_name = "train"
    lmon = DataLoader(dmon, batch_size=256, num_workers=12, pin_memory=True)
    return ltr, lmon, lte, cls_count, mon_name


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    for x, yb in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy()); ys.append(yb.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return accuracy_score(y, p), f1_score(y, p, average="macro"), y, p


def run(model, size, tag, epochs=40, bs=256, lr=3e-3, wd=5e-2, gamma=1.5,
        workers=24, seed=42, gpu=0, pct_start=0.1, model_kwargs=None,
        data_dir=OUT, out_dir="outputs"):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = f"cuda:{gpu}"
    os.makedirs(out_dir, exist_ok=True)

    ltr, lmon, lte, cls_count, mon = make_loaders(size, bs, workers, seed, data_dir)
    classes = json.load(open(f"{data_dir}/classes.json"))
    print(f"[{tag}] model={model} size={size}px  data={data_dir}  "
          f"best-by={mon}-set  train class counts:",
          dict(zip(classes, cls_count.tolist())))

    net = build_model(model, **(model_kwargs or {})).to(device)
    net = net.to(memory_format=torch.channels_last)
    ema = EMA(net, decay=0.999)

    beta = 0.999
    eff = (1 - beta) / (1 - beta ** cls_count)
    w = np.clip(eff / eff.sum() * len(cls_count), 0.3, 5.0)
    crit = FocalLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                     gamma=gamma, smoothing=0.05).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = len(ltr) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=pct_start,
        div_factor=25, final_div_factor=1e3)

    best_f1, best_state, log = 0.0, None, []
    for ep in range(epochs):
        net.train(); t0 = time.time(); run_loss = 0.0
        for x, yb in ltr:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(net(x), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step(); ema.update(net); run_loss += loss.item()
        acc, f1, *_ = evaluate(ema.shadow, lmon, device)
        log.append({"ep": ep, "loss": run_loss / len(ltr),
                    f"{mon}_acc": acc, f"{mon}_f1": f1})
        print(f"[{tag}] ep {ep:02d}  loss {run_loss/len(ltr):.4f}  "
              f"{mon}_acc {acc:.4f}  {mon}_macroF1 {f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(ema.shadow.state_dict())
            torch.save({"model": best_state,
                        "args": {"model": model, "size": size, "tag": tag,
                                 "model_kwargs": model_kwargs or {}},
                        "classes": classes}, f"{out_dir}/best_{tag}.pt")

    # ---- final test with best EMA weights ----
    ema.shadow.load_state_dict(best_state)
    acc, f1, y, p = evaluate(ema.shadow, lte, device)
    mcc = matthews_corrcoef(y, p)
    per = f1_score(y, p, average=None)
    print(f"\n=== [{tag}] TEST  acc {acc:.4f}  macroF1 {f1:.4f}  MCC {mcc:.4f} ===")
    for c, fv in zip(classes, per):
        print(f"  {c:12s} f1 {fv:.4f}")
    json.dump({"log": log, "model": model, "size": size, "best_by": mon,
               "test_acc": acc, "test_macroF1": f1, "test_mcc": mcc,
               "per_class_f1": dict(zip(classes, per.tolist())),
               "confusion": confusion_matrix(y, p).tolist()},
              open(f"{out_dir}/metrics_{tag}.json", "w"), indent=2)
    print(f"[{tag}] saved {out_dir}/best_{tag}.pt and metrics_{tag}.json")
