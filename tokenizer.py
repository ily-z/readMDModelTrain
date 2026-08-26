"""BPE Tokenizer Training for MD-Deployer.

Trains a byte-level BPE tokenizer on markdown + JSON corpus.
Vocab size: 8192 (optimal for 37M param model on narrow domain).
"""

import os
import glob
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def collect_training_texts(data_dir: str) -> list:
    """Collect all text files from data directory for tokenizer training."""
    texts = []
    patterns = ["*.md", "*.txt", "*.json", "*.py", "*.js", "*.ts", "*.yaml", "*.yml"]
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(data_dir, "**", pattern), recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            except Exception:
                continue
    return texts


def train_tokenizer(
    texts: list,
    vocab_size: int = 8192,
    output_path: str = "tokenizer.json",
) -> Tokenizer:
    """Train byte-level BPE tokenizer on markdown+JSON corpus.

    Args:
        texts: List of training text strings.
        vocab_size: Target vocabulary size (default 8192).
        output_path: Where to save the trained tokenizer.

    Returns:
        Trained tokenizer instance.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<sot>", "<eot>", "<pad>", "<unk>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )

    print(f"Training tokenizer on {len(texts)} texts, vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
    """Load a trained tokenizer from disk."""
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    output = sys.argv[3] if len(sys.argv) > 3 else "tokenizer.json"

    texts = collect_training_texts(data_dir)
    if not texts:
        print(f"No texts found in {data_dir}. Add .md, .json, .py etc. files there.")
        sys.exit(1)
    train_tokenizer(texts, vocab_size, output)
