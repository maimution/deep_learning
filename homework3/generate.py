import argparse
from pathlib import Path
from typing import Dict, List

import torch

from models import CharLSTM, TransformerDecoderLM
from poetry_data import BOS, EOS, PAD, UNK, encode, is_seven_char_poem


POEM_FORMATS = {
    "qijue": {"line_count": 4, "max_new_tokens": 40, "min_new_tokens": 28},
    "qilv": {"line_count": 8, "max_new_tokens": 80, "min_new_tokens": 56},
}
BLOCKED_TOKENS = {PAD, BOS, UNK, "□", "{", "}", "…", "\ue802", "="}


def torch_load_checkpoint(path: str | Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(checkpoint: Dict) -> torch.nn.Module:
    model_type = checkpoint["model_type"]
    config = checkpoint["model_config"]
    if model_type == "lstm":
        model = CharLSTM(**config)
    elif model_type == "transformer":
        model = TransformerDecoderLM(**config)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    model.load_state_dict(checkpoint["model_state"])
    return model


def sample_token(
    logits: torch.Tensor,
    token_to_idx: Dict[str, int],
    generated_ids: List[int],
    temperature: float = 0.8,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> int:
    logits = logits.clone()
    for token in BLOCKED_TOKENS:
        if token in token_to_idx:
            logits[token_to_idx[token]] = -float("inf")

    if repetition_penalty > 1.0:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / repetition_penalty
            else:
                logits[token_id] = logits[token_id] * repetition_penalty

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0:
        top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = top_values[-1]
        logits[logits < cutoff] = -float("inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_one(
    model: torch.nn.Module,
    token_to_idx: Dict[str, int],
    idx_to_token: List[str],
    prefix: str = "明月",
    max_new_tokens: int = 80,
    min_new_tokens: int = 0,
    temperature: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    device: torch.device | str = "cpu",
) -> str:
    device = torch.device(device)
    model.eval()

    ids = encode([BOS, *list(prefix)], token_to_idx)
    generated = list(prefix)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1]
            if len(ids) - 1 < min_new_tokens and EOS in token_to_idx:
                logits[token_to_idx[EOS]] = -float("inf")
            next_id = sample_token(
                logits,
                token_to_idx,
                ids,
                temperature,
                top_k,
                repetition_penalty,
            )
            next_token = idx_to_token[next_id]
            ids.append(next_id)

            if next_token == EOS:
                break
            if next_token not in {PAD, BOS, UNK}:
                generated.append(next_token)

    return "".join(generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate poems from a trained checkpoint.")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--poem_format", choices=["qijue", "qilv"], default="qijue")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path. If empty, use checkpoints/{model}_{poem_format}_last.pt.",
    )
    parser.add_argument("--prefix", default="明月")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string, for example: cpu, cuda, cuda:0, cuda:3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    format_spec = POEM_FORMATS[args.poem_format]
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = format_spec["max_new_tokens"]
    min_new_tokens = args.min_new_tokens
    if min_new_tokens is None:
        min_new_tokens = format_spec["min_new_tokens"]

    stem = f"{args.model}_{args.poem_format}"
    checkpoint_path = args.checkpoint or f"checkpoints/{stem}_last.pt"
    output_path = args.output or f"outputs/{stem}_samples.txt"
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    token_to_idx = checkpoint["token_to_idx"]
    idx_to_token = checkpoint["idx_to_token"]

    poems = [
        generate_one(
            model,
            token_to_idx,
            idx_to_token,
            prefix=args.prefix,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        for _ in range(args.samples)
    ]
    correct = [is_seven_char_poem(poem, format_spec["line_count"]) for poem in poems]
    accuracy = sum(correct) / max(len(correct), 1)

    lines = [
        f"checkpoint: {checkpoint_path}",
        f"poem_format: {args.poem_format}",
        f"prefix: {args.prefix}",
        f"max_new_tokens: {max_new_tokens}",
        f"min_new_tokens: {min_new_tokens}",
        f"format_accuracy: {sum(correct)}/{len(correct)} = {accuracy:.2%}",
        "",
    ]
    for index, (poem, ok) in enumerate(zip(poems, correct), start=1):
        lines.append(f"{index:02d}. [{'OK' if ok else 'BAD'}] {poem}")

    result = "\n".join(lines)
    print(result)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
