"""Backbone 1/3 — EfficientNet-B1 (efficient depthwise CNN) @ 64px.

    CUDA_VISIBLE_DEVICES=0 python wafer/train_effb1.py
"""
import argparse
from trainer import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="effb1")
    ap.add_argument("--data-dir", dest="data_dir", default="data/processed")
    ap.add_argument("--out-dir", dest="out_dir", default="outputs")
    a = ap.parse_args()
    run(model="efficientnet_b1", size=64, pct_start=0.1, **vars(a))


if __name__ == "__main__":
    main()
