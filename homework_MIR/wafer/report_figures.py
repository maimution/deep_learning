"""Report figures for one heterogeneous-ensemble model set.

Generates three figures (default = the official-split 96px models):
  1. <prefix>_loss.png       training loss curves, one line per backbone
  2. <prefix>_confusion.png  row-normalized confusion matrix of the ENSEMBLE
  3. <prefix>_perclass.png   per-class F1 bars (3 backbones + ensemble) + summary

Single-model per-class F1 is read from the metrics_*.json files (no recompute);
the ensemble is evaluated once (3 models x TTA) on the test split.

    python wafer/report_figures.py            # official 96px set (defaults)
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix

from model import build_model
from evaluate import predict

CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch"]


def fig_loss(metrics, names, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for m, nm, c in zip(metrics, names, colors):
        log = m["log"]
        ep = [r["ep"] for r in log]; loss = [r["loss"] for r in log]
        ax.plot(ep, loss, label=f"{nm} ({len(log)} ep)", color=c, lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss (focal)")
    ax.set_title("Training loss — official split, 96px")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


def plot_cm(cm, path, title):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, label="row-normalized fraction")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


def fig_perclass(single_f1, ens_f1, summary, names, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.2),
                                   gridspec_kw={"width_ratios": [3, 1]})
    series = names + ["Ensemble+TTA"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    data = single_f1 + [ens_f1]                       # list of 9-length arrays
    x = np.arange(9); w = 0.2
    for k, (vals, s, c) in enumerate(zip(data, series, colors)):
        ax1.bar(x + (k - 1.5) * w, vals, w, label=s, color=c)
    ax1.set_xticks(x); ax1.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax1.set_ylabel("F1"); ax1.set_ylim(0, 1.05)
    ax1.set_title("(A) Per-class F1 — official split, 96px")
    ax1.legend(ncol=2, fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # right: summary metrics (acc / macro-F1 / MCC) x 4 series
    labels = ["accuracy", "macro-F1", "MCC"]
    xs = np.arange(3)
    for k, (s, c) in enumerate(zip(series, colors)):
        ax2.bar(xs + (k - 1.5) * w, summary[k], w, color=c)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylim(0.6, 1.0); ax2.set_title("(B) Overall metrics")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs=3,
                    default=["outputs/official_effb1.pt", "outputs/official_convnext.pt",
                             "outputs/official_swin.pt"])
    ap.add_argument("--metrics", nargs=3,
                    default=["outputs/metrics_official_effb1.json",
                             "outputs/metrics_official_convnext.json",
                             "outputs/metrics_official_swin.json"])
    ap.add_argument("--names", nargs=3, default=["EffNet-B1", "ConvNeXt", "Swin"])
    ap.add_argument("--data-dir", dest="data_dir", default="data/processed_official")
    ap.add_argument("--prefix", default="outputs/official")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"

    metrics = [json.load(open(m)) for m in args.metrics]

    # ---- figure 1: loss curves (from json, no recompute) ----
    fig_loss(metrics, args.names, f"{args.prefix}_loss.png")

    # ---- ensemble pass on the test split (for CM + ensemble bars) ----
    y = np.load(f"{args.data_dir}/labeled_y.npy")
    te = np.load(f"{args.data_dir}/split.npz")["test_idx"]; yte = y[te]
    cache, prob = {}, 0
    for cp in args.ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]; size = a.get("size", 64); mk = a.get("model_kwargs", {})
        m = build_model(a["model"], pretrained=False, **mk).to(device)
        m.load_state_dict(ck["model"]); m = m.to(memory_format=torch.channels_last)
        if size not in cache:
            cache[size] = np.load(f"{args.data_dir}/labeled_X_{size}.npy", mmap_mode="r")
        prob = prob + predict(m, np.ascontiguousarray(cache[size][te]), device)
        print("scored", cp)
    ens_pred = (prob / 3).argmax(1)

    # ---- figure 2: ensemble confusion matrix ----
    plot_cm(confusion_matrix(yte, ens_pred), f"{args.prefix}_confusion.png",
            "Ensemble+TTA confusion matrix — official test (row-normalized)")

    # ---- figure 3: per-class F1 + summary bars ----
    single_f1 = [np.array([m["per_class_f1"][c] for c in CLASSES]) for m in metrics]
    ens_f1 = f1_score(yte, ens_pred, average=None)
    ens_acc = accuracy_score(yte, ens_pred)
    ens_mf1 = f1_score(yte, ens_pred, average="macro")
    ens_mcc = matthews_corrcoef(yte, ens_pred)
    # summary[k] = [acc, macroF1, MCC] for series k
    summary = [[m["test_acc"], m["test_macroF1"], m["test_mcc"]] for m in metrics]
    summary.append([ens_acc, ens_mf1, ens_mcc])       # 4 series x 3 metrics
    fig_perclass(single_f1, ens_f1, summary, args.names, f"{args.prefix}_perclass.png")
    print(f"\nensemble: acc {ens_acc:.4f}  macroF1 {ens_mf1:.4f}  MCC {ens_mcc:.4f}")


if __name__ == "__main__":
    main()
