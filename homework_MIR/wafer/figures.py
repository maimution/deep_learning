"""Report figures for the WM-811K project.

Each figure is one function; run with --fig <name> (or 'all').
Labels are kept in English to avoid CJK-font issues in matplotlib.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "outputs"
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch"]

# Exact per-class counts of the dataset's official train/test split (trainTestLabel).
OFFICIAL = {  # class: (train, test)
    "none": (36730, 110701), "Center": (3462, 832), "Donut": (409, 146),
    "Edge-Loc": (2417, 2772), "Edge-Ring": (8554, 1126), "Loc": (1620, 1973),
    "Near-full": (54, 95), "Random": (609, 257), "Scratch": (500, 693),
}


def fig_split_analysis():
    """Two panels showing why the dataset's official split is problematic."""
    tr = np.array([OFFICIAL[c][0] for c in CLASSES], float)
    te = np.array([OFFICIAL[c][1] for c in CLASSES], float)
    train_frac = tr / (tr + te) * 100.0          # % of each class put into train
    train_prior = tr / tr.sum() * 100.0          # class composition of official train
    test_prior = te / te.sum() * 100.0           # class composition of official test
    x = np.arange(len(CLASSES))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # --- Panel A: per-class train fraction is wildly inconsistent ---
    bars = ax1.bar(x, train_frac, color="#4C72B0", width=0.6)
    ax1.axhline(70, color="#C44E52", ls="--", lw=2,
                label="Stratified (ours): 70% for every class")
    ax1.set_xticks(x); ax1.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax1.set_ylabel("% of class assigned to TRAIN")
    ax1.set_ylim(0, 100)
    ax1.set_title("(A) Official split ratio varies 25%–88% across classes\n"
                  "(should be one constant fraction)")
    for b, v in zip(bars, train_frac):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                 ha="center", fontsize=8)
    ax1.legend(loc="lower right")

    # --- Panel B: train vs test class priors differ -> distribution shift ---
    w = 0.4
    ax2.bar(x - w/2, train_prior, w, label="Official TRAIN composition", color="#55A868")
    ax2.bar(x + w/2, test_prior,  w, label="Official TEST composition",  color="#CCB974")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax2.set_ylabel("% of the set (log scale)")
    ax2.set_yscale("log")
    ax2.set_title("(B) TRAIN and TEST have different class priors\n"
                  "(e.g. Edge-Ring 15.7% of train vs 0.95% of test)")
    # annotate the two worst mismatches
    for c in ["Edge-Ring", "none"]:
        i = CLASSES.index(c)
        ax2.annotate(f"{train_prior[i]:.1f}% vs {test_prior[i]:.1f}%",
                     (i, max(train_prior[i], test_prior[i])),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color="#C44E52")
    ax2.legend(loc="upper right")

    fig.suptitle("Why we re-split instead of using the official train/test labels",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{OUTDIR}/split_analysis.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


FIGS = {"split": fig_split_analysis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="all", choices=["all"] + list(FIGS))
    args = ap.parse_args()
    todo = FIGS.values() if args.fig == "all" else [FIGS[args.fig]]
    for f in todo:
        f()


if __name__ == "__main__":
    main()
