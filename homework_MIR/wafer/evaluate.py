"""Evaluate a checkpoint on the test split with test-time augmentation (TTA)."""
import json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, matthews_corrcoef)

from model import build_model

OUT = "data/processed"


def tta_views(x):
    """8 dihedral views (4 rotations x optional hflip)."""
    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        views.append(r)
        views.append(torch.flip(r, dims=(3,)))
    return views


@torch.no_grad()
def predict(model, X, device, bs=512, tta=True):
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    probs = []
    model.eval()
    for (xb,) in dl:
        xb = (xb.float() / 255.0).to(device).to(memory_format=torch.channels_last)
        views = tta_views(xb) if tta else [xb]
        acc = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model(v).softmax(1).float()
        probs.append((acc / len(views)).cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best_b0.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    a = ck["args"]
    size = a.get("size", 64)                      # legacy checkpoints default to 64px
    mk = a.get("model_kwargs", {})                # e.g. img_size=128 for Swin
    model = build_model(a["model"], pretrained=False, **mk).to(device)
    model.load_state_dict(ck["model"])
    model = model.to(memory_format=torch.channels_last)

    X = np.load(f"{OUT}/labeled_X_{size}.npy", mmap_mode="r")
    y = np.load(f"{OUT}/labeled_y.npy")
    te = np.load(f"{OUT}/split.npz")["test_idx"]

    prob = predict(model, np.ascontiguousarray(X[te]), device, tta=not args.no_tta)
    p = prob.argmax(1); yt = y[te]
    acc = accuracy_score(yt, p); f1 = f1_score(yt, p, average="macro")
    mcc = matthews_corrcoef(yt, p)
    print(f"TEST  acc {acc:.4f}  macroF1 {f1:.4f}  MCC {mcc:.4f}  (TTA={not args.no_tta})")
    print(classification_report(yt, p, target_names=classes, digits=4))
    print("confusion matrix (rows=true):")
    cm = confusion_matrix(yt, p)
    print("            " + " ".join(f"{c[:6]:>6}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:12s}" + " ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()
