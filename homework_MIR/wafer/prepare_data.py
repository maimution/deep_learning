"""
Prepare WM-811K wafer maps for training.

The raw pickle stores each wafer as a variable-sized array with values
{0: background, 1: passing die, 2: failing die}.  We encode every wafer as a
fixed-size 3-channel image and cache the result as compact uint8 arrays:

    channel 0 : defect mask   (map == 2)   -> where the failing dies are
    channel 1 : die mask      (map == 1)   -> the passing dies
    channel 2 : wafer region  (map  > 0)   -> the circular wafer disk

The wafer-region channel keeps the disk boundary, which is essential to tell
Edge-Ring / Edge-Loc apart from interior patterns.

Outputs (in data/processed/):
    labeled_X.npy      (N, 3, S, S) uint8 in {0..255}
    labeled_y.npy      (N,)        int64 class index
    split.npz          train_idx / val_idx / test_idx (stratified)
    classes.json       index -> class name
    unlabeled_X.npy    (M, 3, S, S) uint8   (optional, for semi-supervised)
"""
import os, json, argparse
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

PKL = "data/MIR-WM811K/Python/WM811K.pkl"
OUT = "data/processed"

# Fixed class ordering (9 classes).
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def _scalar(x):
    """failureType / trainTestLabel come wrapped in 1-element arrays."""
    if isinstance(x, (list, np.ndarray)):
        return str(x[0]) if len(x) else ""
    return str(x)


def encode_wafer(wm, size):
    """Variable-size wafer map -> (3, size, size) uint8 image."""
    wm = np.asarray(wm)
    if wm.ndim != 2 or wm.size == 0:
        return np.zeros((3, size, size), np.uint8)
    defect = (wm == 2).astype(np.float32)
    die    = (wm == 1).astype(np.float32)
    region = (wm >  0).astype(np.float32)
    chans = []
    # INTER_AREA gives anti-aliased coverage so thin defects (scratches) survive
    # downsampling instead of disappearing as they would with nearest-neighbour.
    for m in (defect, die, region):
        r = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
        chans.append(np.clip(r * 255.0, 0, 255).astype(np.uint8))
    return np.stack(chans, 0)


_SIZE = 64
def _worker(wm):
    return encode_wafer(wm, _SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--split", choices=["stratified", "official"], default="stratified",
                    help="stratified 70/15/15 (default) or the dataset's official "
                         "Training/Test labels (no validation set)")
    ap.add_argument("--outdir", default=OUT,
                    help="where to write the cache (use a separate dir per split)")
    ap.add_argument("--unlabeled", action="store_true",
                    help="also cache the unlabeled wafers (large)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global _SIZE
    _SIZE = args.size
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("loading pickle ...")
    df = pd.read_pickle(PKL)
    df["ft"]  = df["failureType"].map(_scalar)
    df["ttl"] = df["trainTestLabel"].map(_scalar)

    labeled = df[df["ft"].isin(CLASSES)].reset_index(drop=True)
    print(f"labeled wafers: {len(labeled)}")
    y = labeled["ft"].map(CLS2IDX).to_numpy(np.int64)

    # The encoding depends only on (wafer, size) and the wafer order is fixed, so
    # X / y / classes are identical across splits -> reuse via symlink if the
    # canonical cache already exists, otherwise encode now.
    xname = f"labeled_X_{args.size}.npy"
    canon = os.path.join(OUT, xname)
    if outdir != OUT and os.path.exists(canon):
        for f in (xname, "labeled_y.npy", "classes.json"):
            link = os.path.join(outdir, f)
            if not os.path.exists(link):
                os.symlink(os.path.abspath(os.path.join(OUT, f)), link)
        print(f"reused encoding from {OUT} via symlink")
    else:
        maps = list(labeled["waferMap"].values)
        print(f"encoding {len(maps)} wafers @ {args.size}px on {args.workers} workers ...")
        with Pool(args.workers) as p:
            X = np.stack(p.map(_worker, maps, chunksize=256), 0)
        print("labeled X:", X.shape, X.dtype, "y:", y.shape)
        np.save(os.path.join(outdir, xname), X)
        np.save(os.path.join(outdir, "labeled_y.npy"), y)
        with open(os.path.join(outdir, "classes.json"), "w") as f:
            json.dump(CLASSES, f, indent=2)

    if args.split == "stratified":
        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
        va, te  = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=args.seed)
    else:  # official: Training / Test from the dataset labels, no validation set
        ttl = labeled["ttl"].to_numpy()
        tr = np.where(ttl == "Training")[0]
        te = np.where(ttl == "Test")[0]
        va = np.array([], dtype=int)
    np.savez(os.path.join(outdir, "split.npz"), train_idx=tr, val_idx=va, test_idx=te)
    print(f"[{args.split}] split -> train {len(tr)}  val {len(va)}  test {len(te)}")

    if args.unlabeled:
        unl = df[~df["ft"].isin(CLASSES)].reset_index(drop=True)
        umaps = list(unl["waferMap"].values)
        print(f"encoding {len(umaps)} unlabeled wafers ...")
        with Pool(args.workers) as p:
            UX = np.stack(p.map(_worker, umaps, chunksize=256), 0)
        np.save(os.path.join(outdir, f"unlabeled_X_{args.size}.npy"), UX)
        print("unlabeled X:", UX.shape)

    print("done.")


if __name__ == "__main__":
    main()
