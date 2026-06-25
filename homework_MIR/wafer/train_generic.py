"""Generic single-backbone trainer with a configurable input resolution.

Lets any of the three backbones run at an arbitrary --size (the per-backbone
scripts hardcode their resolution).  Swin gets img_size passed automatically.

    CUDA_VISIBLE_DEVICES=3 python wafer/train_generic.py --model efficientnet_b1 \
        --size 128 --data-dir data/processed_official --out-dir outputs_official_128 --tag effb1
"""
import argparse
from trainer import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--pct-start", dest="pct_start", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", dest="data_dir", default="data/processed")
    ap.add_argument("--out-dir", dest="out_dir", default="outputs")
    a = ap.parse_args()
    mk = {"img_size": a.size} if "swin" in a.model else {}
    run(model=a.model, size=a.size, pct_start=a.pct_start, model_kwargs=mk,
        epochs=a.epochs, bs=a.bs, lr=a.lr, wd=a.wd, gamma=a.gamma,
        workers=a.workers, seed=a.seed, gpu=a.gpu, tag=a.tag,
        data_dir=a.data_dir, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
