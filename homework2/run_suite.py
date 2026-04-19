from __future__ import annotations

import argparse
from pprint import pprint

from svhn_experiments.presets import ABLATION_PRESETS, MODEL_PRESETS, get_ablation_preset, get_model_preset
from svhn_experiments.trainer import train_experiment


def print_banner(title: str) -> None:
    line = "=" * 12
    print(f"\n{line} {title} {line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preset SVHN experiment suites.")
    parser.add_argument("--suite", choices=["backbones", "ablation", "all"], default="all")
    parser.add_argument(
        "--ablation-backbone",
        choices=sorted(MODEL_PRESETS),
        default="se_resnet18",
        help="Backbone used for ablation experiments.",
    )
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="Use a smaller subset for quick checks.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None, help="Override preset epochs for all experiments.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = []

    if args.suite in {"backbones", "all"}:
        for preset_name in ["baseline_cnn", "resnet18", "se_resnet18", "wideresnet", "se_wideresnet"]:
            config = get_model_preset(preset_name)
            config["subset_ratio"] = args.subset_ratio
            config["output_root"] = args.output_root
            config["device"] = args.device
            config["num_workers"] = args.num_workers
            if args.epochs is not None:
                config["epochs"] = args.epochs
            print_banner(f"Running: {config['experiment_name']}")
            summary = train_experiment(config)
            summaries.append(summary)
            print_banner(
                f"Finished: {config['experiment_name']} | best_test_accuracy={summary['best_test_accuracy']:.4f}%"
            )

    if args.suite in {"ablation", "all"}:
        for preset_name in ABLATION_PRESETS:
            config = get_ablation_preset(args.ablation_backbone, preset_name)
            config["subset_ratio"] = args.subset_ratio
            config["output_root"] = args.output_root
            config["device"] = args.device
            config["num_workers"] = args.num_workers
            if args.epochs is not None:
                config["epochs"] = args.epochs
            print_banner(f"Running: {config['experiment_name']}")
            summary = train_experiment(config)
            summaries.append(summary)
            print_banner(
                f"Finished: {config['experiment_name']} | best_test_accuracy={summary['best_test_accuracy']:.4f}%"
            )

    print("\nSummary:")
    for summary in summaries:
        pprint(summary)


if __name__ == "__main__":
    main()
