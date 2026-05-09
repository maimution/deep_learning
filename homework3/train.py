import argparse
import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import CharLSTM, TransformerDecoderLM
from poetry_data import (
    PAD,
    PoetryDataset,
    build_vocab,
    collate_batch,
    load_seven_char_poems,
)


POEM_FORMATS = {
    "qijue": {"line_count": 4, "name": "seven-character quatrains"},
    "qilv": {"line_count": 8, "name": "seven-character regulated verses"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    vocab_size: int,
    pad_id: int,
    args: argparse.Namespace,
) -> Tuple[nn.Module, Dict]:
    if model_type == "lstm":
        config = {
            "vocab_size": vocab_size,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "pad_id": pad_id,
        }
        return CharLSTM(**config), config

    if model_type == "transformer":
        config = {
            "vocab_size": vocab_size,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "feedforward_dim": args.feedforward_dim,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
            "pad_id": pad_id,
        }
        return TransformerDecoderLM(**config), config

    raise ValueError(f"unknown model type: {model_type}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        token_count = target_ids.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def save_loss_plot(
    train_losses: List[float],
    model_type: str,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip plotting loss figure.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy loss")
    plt.title(f"{model_type.upper()} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_loss.png", dpi=160)
    plt.close()


def train_one_model(
    model_type: str,
    train_loader: DataLoader,
    vocab: Dict,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    pad_id = vocab["token_to_idx"][PAD]
    model, model_config = build_model(model_type, len(vocab["idx_to_token"]), pad_id, args)
    model.to(device)

    lr = args.lr_transformer if model_type == "transformer" else args.lr_lstm
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=args.label_smoothing,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses: List[float] = []

    print(f"\nTraining {model_type} ({args.poem_format}) on {device}...")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        train_losses.append(train_loss)

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"{model_type} epoch {epoch:03d}/{args.epochs}: "
                f"train_loss={train_loss:.4f}"
            )

    torch.save(
        {
            "model_type": model_type,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "token_to_idx": vocab["token_to_idx"],
            "idx_to_token": vocab["idx_to_token"],
            "train_losses": train_losses,
            "epoch": args.epochs,
            "poem_format": args.poem_format,
        },
        checkpoint_dir / checkpoint_filename(model_type, args.poem_format),
    )
    save_loss_plot(train_losses, output_stem(model_type, args.poem_format), output_dir)


def output_stem(model_type: str, poem_format: str) -> str:
    return f"{model_type}_{poem_format}"


def checkpoint_filename(model_type: str, poem_format: str) -> str:
    return f"{output_stem(model_type, poem_format)}_last.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM and Transformer poem generators.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="transformer")
    parser.add_argument("--poem_format", choices=["qijue", "qilv"], default="qilv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--feedforward_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr_lstm", type=float, default=2e-3)
    parser.add_argument("--lr_transformer", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string, for example: cpu, cuda, cuda:0, cuda:3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    format_spec = POEM_FORMATS[args.poem_format]
    poems = load_seven_char_poems(args.data_dir, line_count=format_spec["line_count"])
    if not poems:
        raise RuntimeError(f"no {format_spec['name']} found in {args.data_dir}")

    token_to_idx, idx_to_token = build_vocab(poems)
    vocab = {"token_to_idx": token_to_idx, "idx_to_token": idx_to_token}

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_name = f"dataset_summary_{args.poem_format}.json"
    with (Path(args.output_dir) / summary_name).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "poem_format": args.poem_format,
                "total_poems": len(poems),
                "train_poems": len(poems),
                "val_poems": 0,
                "vocab_size": len(idx_to_token),
                "format": f"<BOS> + {format_spec['line_count']} lines x 7 chars, punctuation kept + <EOS>",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pad_id = token_to_idx[PAD]
    train_dataset = PoetryDataset(poems, token_to_idx)
    collate_fn = partial(collate_batch, pad_id=pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(
        f"Loaded {len(poems)} {format_spec['name']} "
        f"(all used for training), vocab={len(idx_to_token)}."
    )

    model_types = ["lstm", "transformer"] if args.model == "both" else [args.model]
    for model_type in model_types:
        train_one_model(model_type, train_loader, vocab, args)


if __name__ == "__main__":
    main()
