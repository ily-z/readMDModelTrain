"""MD-Deployer: 3-Phase Training Loop.

Phase 1: Pretrain on raw text (markdown docs, code, general text)
Phase 2: Distill from teacher on markdown -> manifest pairs
Phase 3: SFT on curated markdown -> deployment manifest pairs
"""

import os
import json
import math
import time
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import ModelConfig, MDTransformer


class TokenDataset(Dataset):
    """Tokenized dataset for pretraining, distillation, and SFT.

    Reads .jsonl files (chat format with "messages" key) and .txt files
    (raw text). Tokenizes using a tokenizer.json file from the tokenizers
    library. Chops data into block_size chunks.
    """

    def __init__(self, data_paths, tokenizer, block_size, mode="pretrain"):
        self.block_size = block_size
        self.mode = mode
        self.tokenizer = tokenizer
        self.token_ids = []

        for data_path in data_paths:
            data_path = Path(data_path)
            if not data_path.exists():
                print(f"Warning: {data_path} does not exist, skipping")
                continue

            if data_path.suffix == ".jsonl":
                self._load_jsonl(data_path)
            elif data_path.suffix == ".txt":
                self._load_txt(data_path)
            elif data_path.is_dir():
                self._load_directory(data_path)

        # Chop into block_size chunks
        self.chunks = []
        for i in range(0, len(self.token_ids) - block_size, block_size):
            chunk = self.token_ids[i : i + block_size + 1]
            if len(chunk) == block_size + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(
            "TokenDataset [{}]: {} tokens, {} chunks".format(
                mode, len(self.token_ids), len(self.chunks)
            )
        )

    def _load_jsonl(self, path):
        """Load .jsonl file with chat format (messages key)."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                messages = data.get("messages", [])
                if not messages:
                    continue

                text = self._format_chat(messages)
                encoded = self.tokenizer.encode(text)
                ids = encoded.ids
                self.token_ids.extend(ids)

    def _load_txt(self, path):
        """Load raw .txt file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        self.token_ids.extend(ids)

    def _load_directory(self, dir_path):
        """Load all .jsonl and .txt files from a directory."""
        for fp in sorted(dir_path.glob("**/*.jsonl")):
            self._load_jsonl(fp)
        for fp in sorted(dir_path.glob("**/*.txt")):
            self._load_txt(fp)

    def _format_chat(self, messages):
        """Format messages using chat template.

        Template: <|start|>{role}\n{content}<|end|>
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append("<|start|>" + role + "\n" + content + "<|end|>")
        return "".join(parts)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def cosine_lr(step, max_steps, lr_min, lr_max, warmup_steps):
    """Cosine learning rate schedule with linear warmup.

    Args:
        step: Current training step.
        max_steps: Total training steps.
        lr_min: Minimum learning rate.
        lr_max: Maximum (peak) learning rate.
        warmup_steps: Number of warmup steps.

    Returns:
        Learning rate at the given step.
    """
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    if step >= max_steps:
        return lr_min
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    model, optimizer, scaler, step, loss, config, checkpoint_dir, phase_name
):
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        scaler: The GradScaler state.
        step: Current training step.
        loss: Current loss value.
        config: Model config as dict.
        checkpoint_dir: Directory to save checkpoints.
        phase_name: Name of the current training phase.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir, phase_name + "_step_" + str(step) + ".pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
            "loss": loss,
            "config": config,
        },
        ckpt_path,
    )
    print("Saved checkpoint: " + ckpt_path)
    return ckpt_path


def train(
    model,
    dataset,
    phase_config,
    hardware_config,
    phase_name,
    device="cuda",
):
    """Run a single training phase.

    Args:
        model: The MDTransformer model.
        dataset: TokenDataset instance.
        phase_config: Phase-specific hyperparameters from config.yaml.
        hardware_config: Hardware settings from config.yaml.
        phase_name: Name of this training phase (for logging/checkpoints).
        device: Device to train on.

    Returns:
        Final training loss.
    """
    model.to(device)

    batch_size = phase_config.get("batch_size", 16)
    lr = phase_config.get("learning_rate", 3e-4)
    lr_min = phase_config.get("lr_min", lr * 0.1)
    warmup_steps = phase_config.get("warmup_steps", 0)
    max_steps = phase_config.get("max_steps", 10000)
    weight_decay = phase_config.get("weight_decay", 0.1)
    grad_clip = phase_config.get("grad_clip", 1.0)
    use_bf16 = phase_config.get("bf16", True)

    log_interval = hardware_config.get("log_interval", 100)
    save_interval = hardware_config.get("save_interval", 2000)
    checkpoint_dir = hardware_config.get("checkpoint_dir", "checkpoints")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    scaler = GradScaler(enabled=use_bf16)

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    start_time = time.time()

    print("\n" + "=" * 60)
    print("Phase: " + phase_name)
    print("Steps: " + str(max_steps))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(lr))
    print("=" * 60)

    while step < max_steps:
        epoch += 1
        for x_batch, y_batch in dataloader:
            if step >= max_steps:
                break

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Update learning rate
            current_lr = cosine_lr(step, max_steps, lr_min, lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Forward pass with mixed precision
            with autocast(enabled=use_bf16, dtype=torch.bfloat16):
                logits, loss = model(x_batch, y_batch)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step += 1

            # Logging
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / log_interval
                tokens_per_sec = (
                    batch_size * dataset.chunks[0].shape[0] * log_interval / elapsed
                )
                print(
                    "[{}] step {}/{} | loss {:.4f} | lr {:.2e} | {:.1f} tok/s | {:.0f}s elapsed".format(
                        phase_name,
                        step,
                        max_steps,
                        avg_loss,
                        current_lr,
                        tokens_per_sec,
                        elapsed,
                    )
                )
                running_loss = 0.0
                start_time = time.time()

            # Checkpointing
            if step % save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step,
                    loss.item(),
                    model.config.__dict__,
                    checkpoint_dir,
                    phase_name,
                )

    # Final checkpoint
    final_loss = running_loss / max(log_interval, 1) if running_loss > 0 else loss.item()
    save_checkpoint(
        model,
        optimizer,
        scaler,
        step,
        final_loss,
        model.config.__dict__,
        checkpoint_dir,
        phase_name + "_final",
    )

    print(phase_name + " complete. Final loss: " + str(final_loss))
    return final_loss


def main():
    """Main training entry point.

    Loads config, tokenizer, and runs 3 phases:
    1. Pretrain on raw text
    2. Distill from teacher on markdown -> manifest pairs
    3. SFT on curated markdown -> deployment manifest pairs
    """
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create model
    model_config = ModelConfig.from_yaml(config_path)
    model = MDTransformer(model_config)

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            "tokenizer.json not found at " + tokenizer_path
        )

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    device = config.get("hardware", {}).get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # Phase 1: Pretrain
    pretrain_data_dir = os.path.join(os.path.dirname(__file__), "data", "pretrain")
    pretrain_paths = []
    if os.path.isdir(pretrain_data_dir):
        for fp in Path(pretrain_data_dir).glob("**/*.txt"):
            pretrain_paths.append(str(fp))
        for fp in Path(pretrain_data_dir).glob("**/*.jsonl"):
            pretrain_paths.append(str(fp))

    if pretrain_paths:
        pretrain_dataset = TokenDataset(
            pretrain_paths,
            tokenizer,
            model_config.block_size,
            mode="pretrain",
        )
        pretrain_config = config.get("training", {}).get("pretrain", {})
        train(
            model,
            pretrain_dataset,
            pretrain_config,
            config.get("hardware", {}),
            "pretrain",
            device=device,
        )
    else:
        print("No pretrain data found in " + pretrain_data_dir + ", skipping phase 1")

    # Phase 2: Distill
    distill_data_dir = os.path.join(os.path.dirname(__file__), "data", "distill")
    distill_paths = []
    if os.path.isdir(distill_data_dir):
        for fp in Path(distill_data_dir).glob("**/*.jsonl"):
            distill_paths.append(str(fp))

    if distill_paths:
        distill_dataset = TokenDataset(
            distill_paths,
            tokenizer,
            model_config.block_size,
            mode="distill",
        )
        distill_config = config.get("training", {}).get("distill", {})
        train(
            model,
            distill_dataset,
            distill_config,
            config.get("hardware", {}),
            "distill",
            device=device,
        )
    else:
        print("No distill data found in " + distill_data_dir + ", skipping phase 2")

    # Phase 3: SFT
    sft_data_dir = os.path.join(os.path.dirname(__file__), "data", "sft")
    sft_paths = []
    if os.path.isdir(sft_data_dir):
        for fp in Path(sft_data_dir).glob("**/*.jsonl"):
            sft_paths.append(str(fp))

    if sft_paths:
        sft_dataset = TokenDataset(
            sft_paths,
            tokenizer,
            model_config.block_size,
            mode="sft",
        )
        sft_config = config.get("training", {}).get("sft", {})
        train(
            model,
            sft_dataset,
            sft_config,
            config.get("hardware", {}),
            "sft",
            device=device,
        )
    else:
        print("No SFT data found in " + sft_data_dir + ", skipping phase 3")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
