"""Produce a confusion-matrix figure and a grid of example wafer maps."""
import json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import build_model
from evaluate import predict

OUT = "data/processed"


def plot_cm(cm, classes, path, title):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("wrote", path)


def plot_examples(X, y, classes, path):
    fig, axes = plt.subplots(1, len(classes), figsize=(2 * len(classes), 2.3))
    for c in range(len(classes)):
        idx = np.where(y == c)[0][0]
        # RGB view: defect=red, pass-die=green, wafer-region=blue
        img = X[idx].transpose(1, 2, 0).astype(np.float32) / 255.0
        axes[c].imshow(img)
        axes[c].set_title(classes[c], fontsize=9)
        axes[c].axis("off")
    fig.suptitle("Encoded wafer maps  (R=defect die, G=pass die, B=wafer region)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best_b0_semi.pt")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = build_model(ck["args"]["model"], pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model = model.to(memory_format=torch.channels_last)

    X = np.load(f"{OUT}/labeled_X.npy"); y = np.load(f"{OUT}/labeled_y.npy")
    te = np.load(f"{OUT}/split.npz")["test_idx"]
    prob = predict(model, X[te], device)
    cm = confusion_matrix(y[te], prob.argmax(1))
    plot_cm(cm, classes, "outputs/confusion_matrix.png", "WM-811K test (row-normalized)")
    plot_examples(X, y, classes, "outputs/wafer_examples.png")


if __name__ == "__main__":
    main()
