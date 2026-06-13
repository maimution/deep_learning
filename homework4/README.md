# Homework 4

This directory contains a standalone PTQ static quantization pipeline for the
`baseline_cnn` model from `homework2`.

The script keeps the homework2 result `95.6131%` as an explicit reference by
reading `homework2/outputs/baseline_cnn/summary.json` and validating that value
before running the quantization experiment.

Run from the repository root:

```bash
python homework4/run_quantization.py
```

Main outputs are written to:

- `homework4/outputs/baseline_cnn_ptq/summary.json`
- `homework4/outputs/baseline_cnn_ptq/comparison.csv`
- `homework4/outputs/baseline_cnn_ptq/layer_mse.csv`
- `homework4/outputs/baseline_cnn_ptq/accuracy_comparison.png`
- `homework4/outputs/baseline_cnn_ptq/latency_comparison.png`

Useful options:

```bash
python homework4/run_quantization.py --calibration-size 1024 --error-batches 2
python homework4/run_quantization.py --latency-runs 500 --latency-threads 1
```
