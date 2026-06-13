from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QConfig, QuantStub, convert, fuse_modules, prepare
from torch.ao.quantization.observer import MinMaxObserver
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
HOMEWORK2_ROOT = REPO_ROOT / "homework2"
if str(HOMEWORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(HOMEWORK2_ROOT))

from svhn_experiments.data import build_dataloaders  # noqa: E402
from svhn_experiments.models import BaselineCNN  # noqa: E402
from svhn_experiments.trainer import evaluate, set_seed  # noqa: E402

EXPECTED_REFERENCE_ACCURACY = 95.6131
LAYER_PAIRS = [
    ("block1_relu1", "features.2", "features.0"),
    ("block1_relu2", "features.5", "features.3"),
    ("pool1", "features.6", "features.6"),
    ("block2_relu1", "features.9", "features.7"),
    ("block2_relu2", "features.12", "features.10"),
    ("pool2", "features.13", "features.13"),
    ("block3_relu1", "features.16", "features.14"),
    ("block3_relu2", "features.19", "features.17"),
    ("adaptive_pool", "features.20", "features.20"),
    ("classifier_relu", "classifier.3", "classifier.2"),
    ("logits", "classifier.5", "classifier.5"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PTQ static quantization for the homework2 baseline CNN.")
    parser.add_argument("--train-path", default=str(HOMEWORK2_ROOT / "train_32x32.mat"))
    parser.add_argument("--test-path", default=str(HOMEWORK2_ROOT / "test_32x32.mat"))
    parser.add_argument("--checkpoint-path", default=str(HOMEWORK2_ROOT / "outputs" / "baseline_cnn" / "best.pt"))
    parser.add_argument(
        "--reference-summary-path",
        default=str(HOMEWORK2_ROOT / "outputs" / "baseline_cnn" / "summary.json"),
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "homework4" / "outputs" / "baseline_cnn_ptq"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--calibration-size", type=int, default=2048)
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-runs", type=int, default=300)
    parser.add_argument("--latency-threads", type=int, default=1)
    parser.add_argument("--error-batches", type=int, default=4)
    parser.add_argument("--backend", choices=["x86", "fbgemm", "qnnpack", "onednn"], default="x86")
    return parser


def quant_range(num_bits: int, signed: bool) -> tuple[int, int]:
    if signed:
        return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    return 0, 2**num_bits - 1


def linear_quantize(
    x: torch.Tensor,
    num_bits: int = 8,
    *,
    signed: bool = True,
) -> tuple[torch.Tensor, float, int]:
    qmin, qmax = quant_range(num_bits, signed)
    x_min = float(x.min().item())
    x_max = float(x.max().item())

    if x_min == x_max:
        zero_point = 0 if signed else qmin
        target_dtype = torch.int8 if signed else torch.uint8
        q = torch.full_like(x, fill_value=zero_point, dtype=target_dtype)
        return q, 1.0, int(zero_point)

    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point = qmin - x_min / scale
    zero_point = int(round(max(qmin, min(qmax, zero_point))))

    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    target_dtype = torch.int8 if signed else torch.uint8
    return q.to(dtype=target_dtype), float(scale), zero_point


def linear_dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return scale * (q.to(torch.float32) - float(zero_point))


class QuantizableBaselineCNN(BaselineCNN):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        fuse_modules(
            self.features,
            [
                ["0", "1", "2"],
                ["3", "4", "5"],
                ["7", "8", "9"],
                ["10", "11", "12"],
                ["14", "15", "16"],
                ["17", "18", "19"],
            ],
            inplace=True,
        )
        fuse_modules(self.classifier, [["2", "3"]], inplace=True)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_csv(rows: Iterable[dict[str, object]], fieldnames: list[str], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = module._modules[part]
    return module


def capture_outputs(model: nn.Module, inputs: torch.Tensor, module_names: list[str]) -> OrderedDict[str, torch.Tensor]:
    outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
    hooks = []

    for module_name in module_names:
        module = get_module_by_name(model, module_name)

        def _hook(_module, _args, output, *, key: str = module_name) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.is_quantized:
                tensor = tensor.dequantize()
            outputs[key] = tensor.detach().to(torch.float32).cpu()

        hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return outputs


def load_fp32_model(checkpoint_path: str | Path) -> BaselineCNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BaselineCNN().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_quantizable_model(checkpoint_path: str | Path) -> QuantizableBaselineCNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = QuantizableBaselineCNN().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model


def build_qconfig() -> QConfig:
    return QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
    )


def build_calibration_loader(
    dataset,
    calibration_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    calibration_size = min(calibration_size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:calibration_size].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def calibrate_model(model: nn.Module, calibration_loader: DataLoader) -> None:
    model.eval()
    with torch.inference_mode():
        for inputs, _targets in calibration_loader:
            model(inputs)


def quantize_model(
    checkpoint_path: str | Path,
    calibration_loader: DataLoader,
    backend: str,
) -> nn.Module:
    if backend not in torch.backends.quantized.supported_engines:
        raise ValueError(
            f"Quantized backend {backend!r} is not supported. "
            f"Available backends: {torch.backends.quantized.supported_engines}"
        )

    torch.backends.quantized.engine = backend
    model = load_quantizable_model(checkpoint_path)
    model.fuse_model()
    model.qconfig = build_qconfig()
    prepare(model, inplace=True)
    calibrate_model(model, calibration_loader)
    convert(model, inplace=True)
    return model.eval()


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, torch.device("cpu"), collect_examples=False)
    return {
        "loss": round(float(metrics["loss"]), 6),
        "accuracy": round(float(metrics["accuracy"]), 4),
    }


def save_state_dict_and_measure_size(model: nn.Module, path: str | Path) -> float:
    path = Path(path)
    torch.save(model.state_dict(), path)
    return round(path.stat().st_size / (1024.0 * 1024.0), 4)


def measure_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int,
    runs: int,
    threads: int,
) -> float:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        started_at = time.perf_counter()
        for _ in range(runs):
            model(sample)
        elapsed = time.perf_counter() - started_at

    torch.set_num_threads(previous_threads)
    return round(elapsed * 1000.0 / runs, 6)


def compute_layer_mse(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    loader: DataLoader,
    error_batches: int,
) -> OrderedDict[str, float]:
    float_names = [float_name for _stage, float_name, _int8_name in LAYER_PAIRS]
    int8_names = [int8_name for _stage, _float_name, int8_name in LAYER_PAIRS]
    mse_sum = OrderedDict((stage, 0.0) for stage, _float_name, _int8_name in LAYER_PAIRS)
    mse_count = OrderedDict((stage, 0) for stage, _float_name, _int8_name in LAYER_PAIRS)

    for batch_index, (inputs, _targets) in enumerate(loader):
        if batch_index >= error_batches:
            break
        fp32_outputs = capture_outputs(fp32_model, inputs, float_names)
        int8_outputs = capture_outputs(int8_model, inputs, int8_names)

        for stage, float_name, int8_name in LAYER_PAIRS:
            diff = fp32_outputs[float_name] - int8_outputs[int8_name]
            mse_sum[stage] += float(diff.pow(2).sum().item())
            mse_count[stage] += diff.numel()

    return OrderedDict(
        (stage, round(mse_sum[stage] / max(1, mse_count[stage]), 10))
        for stage, _float_name, _int8_name in LAYER_PAIRS
    )


def manual_quantization_report(
    fp32_model: nn.Module,
    sample_batch: torch.Tensor,
) -> dict[str, float | int]:
    first_conv_weight = fp32_model.features[0].weight.detach().cpu()
    q_weight, weight_scale, weight_zero_point = linear_quantize(first_conv_weight, signed=True)
    dq_weight = linear_dequantize(q_weight, weight_scale, weight_zero_point)
    weight_mse = torch.mean((first_conv_weight - dq_weight) ** 2).item()

    activation = capture_outputs(fp32_model, sample_batch, ["features.2"])["features.2"]
    q_activation, activation_scale, activation_zero_point = linear_quantize(activation, signed=False)
    dq_activation = linear_dequantize(q_activation, activation_scale, activation_zero_point)
    activation_mse = torch.mean((activation - dq_activation) ** 2).item()

    return {
        "first_conv_weight_scale": round(float(weight_scale), 10),
        "first_conv_weight_zero_point": int(weight_zero_point),
        "first_conv_weight_mse": round(float(weight_mse), 10),
        "first_activation_scale": round(float(activation_scale), 10),
        "first_activation_zero_point": int(activation_zero_point),
        "first_activation_mse": round(float(activation_mse), 10),
    }


def plot_comparison(labels: list[str], values: list[float], ylabel: str, title: str, path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#4C72B0", "#DD8452"])
    plt.ylabel(ylabel)
    plt.title(title)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def validate_reference_accuracy(summary_path: str | Path, expected_accuracy: float) -> dict:
    payload = load_json(summary_path)
    reference_accuracy = round(float(payload["best_test_accuracy"]), 4)
    if abs(reference_accuracy - expected_accuracy) > 1e-4:
        raise ValueError(
            f"Expected homework2 baseline reference accuracy {expected_accuracy:.4f}, "
            f"but found {reference_accuracy:.4f} in {summary_path}."
        )
    return payload


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    reference_summary = validate_reference_accuracy(args.reference_summary_path, EXPECTED_REFERENCE_ACCURACY)

    data = build_dataloaders(
        train_path=args.train_path,
        test_path=args.test_path,
        batch_size=args.calibration_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        augmentation="none",
        subset_ratio=1.0,
        seed=args.seed,
    )
    calibration_loader = build_calibration_loader(
        data.train_loader.dataset,
        calibration_size=args.calibration_size,
        batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    fp32_model = load_fp32_model(args.checkpoint_path).eval()
    fp32_metrics = evaluate_model(fp32_model, data.test_loader)
    fp32_size_mb = save_state_dict_and_measure_size(fp32_model, output_dir / "fp32_state_dict.pt")

    sample_batch, _sample_targets = next(iter(data.test_loader))
    latency_sample = sample_batch[:1].contiguous()
    manual_report = manual_quantization_report(fp32_model, sample_batch[: args.calibration_batch_size])
    fp32_latency_ms = measure_latency_ms(
        fp32_model,
        latency_sample,
        warmup=args.latency_warmup,
        runs=args.latency_runs,
        threads=args.latency_threads,
    )

    int8_model = quantize_model(
        checkpoint_path=args.checkpoint_path,
        calibration_loader=calibration_loader,
        backend=args.backend,
    )
    int8_metrics = evaluate_model(int8_model, data.test_loader)
    int8_size_mb = save_state_dict_and_measure_size(int8_model, output_dir / "int8_state_dict.pt")
    int8_latency_ms = measure_latency_ms(
        int8_model,
        latency_sample,
        warmup=args.latency_warmup,
        runs=args.latency_runs,
        threads=args.latency_threads,
    )

    layer_mse = compute_layer_mse(fp32_model, int8_model, data.test_loader, error_batches=args.error_batches)

    accuracy_drop = round(fp32_metrics["accuracy"] - int8_metrics["accuracy"], 4)
    compression_ratio = round(fp32_size_mb / int8_size_mb, 4)
    speedup_ratio = round(fp32_latency_ms / int8_latency_ms, 4)

    summary = {
        "reference_accuracy_from_homework2": round(float(reference_summary["best_test_accuracy"]), 4),
        "measured_fp32_accuracy": fp32_metrics["accuracy"],
        "measured_int8_accuracy": int8_metrics["accuracy"],
        "fp32_test_loss": fp32_metrics["loss"],
        "int8_test_loss": int8_metrics["loss"],
        "accuracy_drop": accuracy_drop,
        "fp32_model_size_mb": fp32_size_mb,
        "int8_model_size_mb": int8_size_mb,
        "compression_ratio": compression_ratio,
        "fp32_latency_ms": fp32_latency_ms,
        "int8_latency_ms": int8_latency_ms,
        "speedup_ratio": speedup_ratio,
        "calibration_size": min(args.calibration_size, data.train_size),
        "latency_runs": args.latency_runs,
        "latency_warmup": args.latency_warmup,
        "latency_threads": args.latency_threads,
        "quant_backend": args.backend,
        "manual_quantization": manual_report,
        "per_layer_mse": layer_mse,
        "source_checkpoint": str(Path(args.checkpoint_path).resolve()),
        "source_reference_summary": str(Path(args.reference_summary_path).resolve()),
    }

    save_json(summary, output_dir / "summary.json")
    save_csv(
        [
            {"model": "fp32", "accuracy": fp32_metrics["accuracy"], "model_size_mb": fp32_size_mb, "latency_ms": fp32_latency_ms},
            {"model": "int8", "accuracy": int8_metrics["accuracy"], "model_size_mb": int8_size_mb, "latency_ms": int8_latency_ms},
        ],
        fieldnames=["model", "accuracy", "model_size_mb", "latency_ms"],
        path=output_dir / "comparison.csv",
    )
    save_csv(
        [{"layer": layer_name, "mse": mse_value} for layer_name, mse_value in layer_mse.items()],
        fieldnames=["layer", "mse"],
        path=output_dir / "layer_mse.csv",
    )

    plot_comparison(
        labels=["FP32", "INT8"],
        values=[fp32_metrics["accuracy"], int8_metrics["accuracy"]],
        ylabel="Test Accuracy (%)",
        title="SVHN Accuracy Comparison",
        path=output_dir / "accuracy_comparison.png",
    )
    plot_comparison(
        labels=["FP32", "INT8"],
        values=[fp32_latency_ms, int8_latency_ms],
        ylabel="Single-image Latency (ms)",
        title="SVHN Latency Comparison",
        path=output_dir / "latency_comparison.png",
    )

    print("Saved outputs to:", output_dir)
    pprint(summary)


if __name__ == "__main__":
    main()
