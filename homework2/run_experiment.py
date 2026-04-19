from __future__ import annotations

import argparse
from pprint import pprint

from svhn_experiments.presets import DEFAULT_CONFIG, MODEL_PRESETS, get_model_preset, merge_config
from svhn_experiments.trainer import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SVHN image classification models.")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default=None, help="Named model preset.")
    parser.add_argument("--train-path", default=DEFAULT_CONFIG["train_path"])
    parser.add_argument("--test-path", default=DEFAULT_CONFIG["test_path"])
    parser.add_argument("--output-root", default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--experiment-name", default=DEFAULT_CONFIG["experiment_name"])
    parser.add_argument(
        "--model",
        choices=["baseline_cnn", "resnet18", "se_resnet18", "wideresnet", "se_wideresnet"],
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mixup-alpha", type=float, default=None)
    parser.add_argument("--augmentation", choices=["none", "crop", "standard"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--se-reduction", type=int, default=None)
    parser.add_argument("--wide-depth", type=int, default=None)
    parser.add_argument("--wide-factor", type=int, default=None)
    parser.add_argument("--subset-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser


def namespace_to_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    if args.preset:
        config = get_model_preset(args.preset)

    overrides = {
        "train_path": args.train_path,
        "test_path": args.test_path,
        "output_root": args.output_root,
        "experiment_name": args.experiment_name,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "augmentation": args.augmentation,
        "dropout": args.dropout,
        "se_reduction": args.se_reduction,
        "wide_depth": args.wide_depth,
        "wide_factor": args.wide_factor,
        "subset_ratio": args.subset_ratio,
        "seed": args.seed,
        "device": args.device,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return merge_config(config, overrides)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = namespace_to_config(args)
    pprint(config)
    summary = train_experiment(config)
    print("\nFinished experiment:")
    pprint(summary)


if __name__ == "__main__":
    main()
