"""t-SNE of the official 3-model ensemble feature space.

Concatenates the penultimate (pre-logit) features of the three official-split
backbones -> one 2816-d "ensemble representation" per wafer, then PCA(50)+t-SNE
to 2D, colored by true class.  Uses a class-balanced subsample so rare classes
are visible and t-SNE stays fast.

    python wafer/tsne_figure.py
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model import build_model

CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch"]


@torch.no_grad()
def feats(model, X, device, bs=512):
    out = []
    for s in range(0, len(X), bs):
        x = torch.from_numpy(X[s:s + bs]).float().div(255).to(device).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs=3,
                    default=["outputs/official_effb1.pt", "outputs/official_convnext.pt",
                             "outputs/official_swin.pt"])
    ap.add_argument("--data-dir", dest="data_dir", default="data/processed_official")
    ap.add_argument("--cap", type=int, default=800, help="max samples per class")
    ap.add_argument("--out", default="figure/official_tsne.png")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    rng = np.random.default_rng(args.seed)

    y = np.load(f"{args.data_dir}/labeled_y.npy")
    te = np.load(f"{args.data_dir}/split.npz")["test_idx"]
    yte = y[te]
    # class-balanced subsample of the test set
    pick = []
    for c in range(9):
        idx = np.where(yte == c)[0]
        if len(idx) > args.cap:
            idx = rng.choice(idx, args.cap, replace=False)
        pick.append(idx)
    pick = np.sort(np.concatenate(pick))
    ys = yte[pick]
    print(f"subsample {len(pick)} wafers ({dict(zip(CLASSES, np.bincount(ys, minlength=9)))})")

    # extract + concat penultimate features from the 3 backbones
    cache, parts = {}, []
    for cp in args.ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]; size = a.get("size", 96); mk = a.get("model_kwargs", {})
        m = build_model(a["model"], pretrained=False, **mk).to(device).eval()
        m.load_state_dict(ck["model"]); m = m.to(memory_format=torch.channels_last)
        if size not in cache:
            cache[size] = np.load(f"{args.data_dir}/labeled_X_{size}.npy", mmap_mode="r")
        Xs = np.ascontiguousarray(cache[size][te[pick]])
        f = feats(m, Xs, device)
        parts.append(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-6))  # L2-norm per model
        print(f"  {a['model']:32s} feat {f.shape}")
    F = np.concatenate(parts, 1)
    print("concatenated feature:", F.shape)

    Z = PCA(n_components=50, random_state=args.seed).fit_transform(F)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed,
               metric="cosine").fit_transform(Z)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10")
    for c in range(9):
        m = ys == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(c), label=CLASSES[c], alpha=0.7,
                   edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE of the 3-model ensemble features — official split\n"
                 "(penultimate features of EffNet-B1 + ConvNeXt + Swin, concatenated)")
    ax.legend(markerscale=2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(); fig.savefig(args.out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
