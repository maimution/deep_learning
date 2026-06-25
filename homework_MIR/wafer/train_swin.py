"""Backbone 3/3 — Swin-Tiny (window self-attention transformer) @ 128px.

Fed a higher resolution than the CNNs: at 128px Swin gets 32x32 patch tokens,
enough for window attention to be meaningful (64px would be too few tokens).
Transformers are data-hungry, so a gentler LR and longer warmup than the CNNs.

    CUDA_VISIBLE_DEVICES=2 python wafer/train_swin.py
"""
import argparse
from trainer import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)      # 128px -> smaller batch
    ap.add_argument("--lr", type=float, default=8e-4)   # transformers want lower LR
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="swin")
    ap.add_argument("--data-dir", dest="data_dir", default="data/processed")
    ap.add_argument("--out-dir", dest="out_dir", default="outputs")
    a = ap.parse_args()
    run(model="swin_tiny_patch4_window7_224", size=128, pct_start=0.2,
        model_kwargs={"img_size": 128}, **vars(a))


if __name__ == "__main__":
    main()
