import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def _split_poem_lines(paragraphs: Sequence[str]) -> List[str]:
    """Split paragraph/couplet strings into punctuation-free sentence strings."""
    lines: List[str] = []
    for paragraph in paragraphs:
        for part in re.split(r"[，。！？；]", paragraph):
            line = re.sub(r"[、：「」『』（）《》〈〉\[\]\s]", "", part)
            if line:
                lines.append(line)
    return lines


def normalize_jueju(lines: Sequence[str]) -> str:
    return f"{lines[0]}，{lines[1]}。{lines[2]}，{lines[3]}。"


def normalize_seven_char_poem(lines: Sequence[str]) -> str:
    poem = []
    for index, line in enumerate(lines):
        poem.append(line)
        poem.append("，" if index % 2 == 0 else "。")
    return "".join(poem)


def load_seven_char_poems(data_dir: str | Path, line_count: int) -> List[str]:
    data_path = Path(data_dir)
    poems: List[str] = []

    for json_file in sorted(data_path.glob("poet.song.*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            records = json.load(f)

        for item in records:
            lines = _split_poem_lines(item.get("paragraphs", []))
            if len(lines) == line_count and all(len(line) == 7 for line in lines):
                poems.append(normalize_seven_char_poem(lines))

    return poems


def load_seven_char_quatrains(data_dir: str | Path) -> List[str]:
    return load_seven_char_poems(data_dir, line_count=4)


def load_seven_char_lushi(data_dir: str | Path) -> List[str]:
    return load_seven_char_poems(data_dir, line_count=8)


def split_train_val(
    poems: Sequence[str], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str]]:
    poems = list(poems)
    rng = random.Random(seed)
    rng.shuffle(poems)
    val_size = max(1, int(len(poems) * val_ratio))
    return poems[val_size:], poems[:val_size]


def build_vocab(poems: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    chars = sorted({char for poem in poems for char in poem})
    idx_to_token = SPECIAL_TOKENS + chars
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    return token_to_idx, idx_to_token


def encode(tokens: Sequence[str], token_to_idx: Dict[str, int]) -> List[int]:
    unk_id = token_to_idx[UNK]
    return [token_to_idx.get(token, unk_id) for token in tokens]


def decode(ids: Sequence[int], idx_to_token: Sequence[str]) -> str:
    tokens = []
    for idx in ids:
        token = idx_to_token[int(idx)]
        if token == EOS:
            break
        if token in {PAD, BOS, UNK}:
            continue
        tokens.append(token)
    return "".join(tokens)


class PoetryDataset(Dataset):
    def __init__(self, poems: Sequence[str], token_to_idx: Dict[str, int]):
        self.samples = []
        for poem in poems:
            token_ids = encode([BOS, *list(poem), EOS], token_to_idx)
            self.samples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.samples[index]
        return ids[:-1], ids[1:]


def collate_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    batch_size = len(batch)

    input_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    target_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for row, (input_ids, target_ids) in enumerate(zip(inputs, targets)):
        input_batch[row, : input_ids.size(0)] = input_ids
        target_batch[row, : target_ids.size(0)] = target_ids

    return input_batch, target_batch


def is_seven_char_quatrain(text: str) -> bool:
    return bool(re.fullmatch(r"[^，。！？；\s]{7}，[^，。！？；\s]{7}。[^，。！？；\s]{7}，[^，。！？；\s]{7}。", text))


def is_seven_char_poem(text: str, line_count: int) -> bool:
    pattern = "".join(
        r"[^，。！？；\s]{7}" + ("，" if index % 2 == 0 else "。")
        for index in range(line_count)
    )
    return bool(re.fullmatch(pattern, text))
