"""Ensemble of multiple checkpoints with TTA, evaluated on the test split.

Supports heterogeneous backbones at different input resolutions: each checkpoint
records its own `size` (and any `model_kwargs`, e.g. img_size for Swin), so a
64px CNN and a 128px transformer can vote together.  All models see the *same*
test wafers (the split is shared across resolutions), so their softmax outputs
align row-for-row and can be averaged directly.
"""
import argparse
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, matthews_corrcoef)

from model import build_model
from evaluate import predict

OUT = "data/processed"


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    size = a.get("size", 64)                  # legacy checkpoints default to 64px
    mk = a.get("model_kwargs", {})
    m = build_model(a["model"], pretrained=False, **mk).to(device)
    m.load_state_dict(ck["model"])
    return m.to(memory_format=torch.channels_last), ck["classes"], size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+",
                    default=["outputs/best_effb1.pt", "outputs/best_convnext.pt",
                             "outputs/best_swin.pt"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-dir", dest="data_dir", default=OUT)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dd = args.data_dir

    y = np.load(f"{dd}/labeled_y.npy")
    te = np.load(f"{dd}/split.npz")["test_idx"]
    yte = y[te]

    cache = {}                                # avoid reloading the same-size array
    prob, classes = 0, None
    for c in args.ckpts:
        m, classes, size = load(c, device)
        if size not in cache:
            cache[size] = np.load(f"{dd}/labeled_X_{size}.npy", mmap_mode="r")
        Xte = np.ascontiguousarray(cache[size][te])
        prob = prob + predict(m, Xte, device)      # averaged softmax (with TTA)
        print(f"scored {c}  (size {size}px)")
    pred = (prob / len(args.ckpts)).argmax(1)

    acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")
    mcc = matthews_corrcoef(yte, pred)
    print(f"\nENSEMBLE+TTA  acc {acc:.4f}  macroF1 {f1:.4f}  MCC {mcc:.4f}  "
          f"({len(args.ckpts)} models)")
    print(classification_report(yte, pred, target_names=classes, digits=4))


if __name__ == "__main__":
    main()
