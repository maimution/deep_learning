"""Backbone 2/3 — ConvNeXt-Tiny (large-kernel modern CNN) @ 64px.

    CUDA_VISIBLE_DEVICES=1 python wafer/train_convnext.py
"""
import argparse
from trainer import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)   # ConvNeXt likes a gentler LR
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="convnext")
    ap.add_argument("--data-dir", dest="data_dir", default="data/processed")
    ap.add_argument("--out-dir", dest="out_dir", default="outputs")
    a = ap.parse_args()
    run(model="convnext_tiny", size=64, pct_start=0.1, **vars(a))


if __name__ == "__main__":
    main()
