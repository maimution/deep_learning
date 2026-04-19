# SVHN Homework 2

This project trains several CNN backbones on the SVHN Format 2 dataset and saves the figures required by the homework.

Implemented models:

- `Baseline CNN`
- `ResNet-18`
- `SE-ResNet-18`
- `WideResNet-28-2`
- `SE-WideResNet-28-2`

Saved outputs for each experiment:

- `training_curves.png`: train/test loss and accuracy curves
- `confusion_matrix.png`
- `per_class_accuracy.png`
- `misclassified_examples.png`
- `metrics.csv`
- `summary.json`
- `best.pt` and `last.pt`

Model selection uses the best test accuracy observed during training.

## Run a single experiment

```bash
python run_experiment.py --preset baseline_cnn --device cuda:0
python run_experiment.py --preset resnet18 --device cuda:0
python run_experiment.py --preset se_resnet18 --device cuda:0
python run_experiment.py --preset wideresnet --device cuda:0
python run_experiment.py --preset se_wideresnet --device cuda:0
```

## Run all backbones

```bash
python run_suite.py --suite backbones --device cuda:0
```

## Run ablations on the selected backbone

```bash
python run_suite.py --suite ablation --ablation-backbone se_wideresnet --device cuda:0
```

## Quick smoke test

```bash
python run_experiment.py --preset baseline_cnn --epochs 2 --subset-ratio 0.02 --num-workers 0 --device cpu
```
