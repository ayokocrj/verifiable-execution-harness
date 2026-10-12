# -*- coding: utf-8 -*-
"""
PyTorch Training Script for the Ledger-Invariant decision model family.

v0.1 (default): 10M config, binary ALLOW / BLOCK gatekeeper, traces only.
    python scripts/train.py --model_size 10m --num_classes 2 \
        --dataset data/synthetic_traces_50k.jsonl --epochs 5 --batch_size 64

v0.2: 35m config, three classes (adds ROLLBACK).
    python scripts/train.py --model_size 35m --num_classes 3 --dataset data/synthetic_traces_v02.jsonl

Outputs (in --checkpoint_dir):
    ledger_invariant_<size>_final.pt   state_dict
    config.json                        architecture config (loaded back by DecisionEngine)
    metrics.json                       val accuracy, per-class recall, confusion matrix, held-out test metrics
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.tokenizer import InvariantTokenizer
from models.architecture import LedgerInvariantConfig, LedgerInvariantDecisionTransformer

LABELS = ["ALLOW", "BLOCK", "ROLLBACK"]


def load_jsonl(path: str, num_classes: int) -> List[Dict[str, Any]]:
    """Loads traces and keeps only those whose class_id fits the model head."""
    samples = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["class_id"] < num_classes:
                samples.append(item)
            else:
                dropped += 1
    if dropped:
        print(f"[!] Dropped {dropped:,} samples whose class_id >= num_classes={num_classes} (e.g. ROLLBACK in v0.1).")
    return samples


if TORCH_AVAILABLE:
    class InvariantTraceDataset(Dataset):
        def __init__(self, samples: List[Dict[str, Any]], tokenizer: InvariantTokenizer, max_len: int):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.pad_id = tokenizer.token_to_id["[PAD]"]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            text = self.tokenizer.format_input(
                item["state"], item["action_name"], item["action_params"], item["invariants"]
            )
            token_ids = self.tokenizer.encode(text, max_length=self.max_len)
            padded = self.tokenizer.pad(token_ids, max_length=self.max_len)
            return torch.tensor(padded, dtype=torch.long), torch.tensor(item["class_id"], dtype=torch.long)

    def padding_mask_from(inputs: "torch.Tensor", pad_id: int) -> "torch.Tensor":
        """True where the token is [PAD]; matches nn.MultiheadAttention key_padding_mask semantics."""
        return inputs.eq(pad_id)

    @torch.no_grad()
    def evaluate(model, loader, device, pad_id: int, num_classes: int) -> Dict[str, Any]:
        model.eval()
        confusion = [[0] * num_classes for _ in range(num_classes)]
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, padding_mask=padding_mask_from(inputs, pad_id))
            preds = torch.argmax(logits, dim=-1)
            for t, p in zip(targets.tolist(), preds.tolist()):
                confusion[t][p] += 1
        total = sum(sum(row) for row in confusion)
        correct = sum(confusion[i][i] for i in range(num_classes))
        per_class_recall = {
            LABELS[i]: (confusion[i][i] / sum(confusion[i])) if sum(confusion[i]) else None
            for i in range(num_classes)
        }
        return {
            "accuracy": correct / total if total else 0.0,
            "per_class_recall": per_class_recall,
            "confusion_matrix": confusion,
            "n": total,
        }


def train(args):
    if not TORCH_AVAILABLE:
        print("[!] PyTorch is not installed in the local environment. Please run on GPU (Colab/RunPod).")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_size == "35m":
        config = LedgerInvariantConfig.get_35m_config(num_classes=args.num_classes)
    else:
        config = LedgerInvariantConfig.get_10m_config(num_classes=args.num_classes)
    tokenizer = InvariantTokenizer(vocab_size=config.vocab_size)
    pad_id = tokenizer.token_to_id["[PAD]"]

    print(f"[*] Initializing model {args.model_size.upper()} ({config.count_parameters():,} parameters, "
          f"{config.num_classes} classes: {LABELS[:config.num_classes]})...")
    model = LedgerInvariantDecisionTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[*] Training on device: {device}")

    if not os.path.exists(args.dataset):
        print(f"[!] Dataset {args.dataset} not found. Generate it first using scripts/generate_dataset.py.")
        return

    samples = load_jsonl(args.dataset, config.num_classes)
    dataset = InvariantTraceDataset(samples, tokenizer, max_len=config.max_seq_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = []

    print(f"[*] Starting training loop for {args.epochs} epochs on {len(train_ds):,} train / {len(val_ds):,} val samples...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs, padding_mask=padding_mask_from(inputs, pad_id))
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        val = evaluate(model, val_loader, device, pad_id, config.num_classes)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_accuracy": val["accuracy"]})
        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_loss:.4f} - Val Accuracy: {val['accuracy'] * 100:.2f}%")

    out_weights = os.path.join(args.checkpoint_dir, f"ledger_invariant_{args.model_size}_final.pt")
    torch.save(model.state_dict(), out_weights)
    config.save(os.path.join(args.checkpoint_dir, "config.json"))
    print(f"[+] Model weights saved to {out_weights} (config.json alongside)")

    metrics: Dict[str, Any] = {
        "model_size": args.model_size,
        "num_classes": config.num_classes,
        "labels": LABELS[:config.num_classes],
        "parameters": config.count_parameters(),
        "seed": args.seed,
        "history": history,
        "validation": evaluate(model, val_loader, device, pad_id, config.num_classes),
    }

    # Held-out test set (never seen in training) if available
    if args.test_dataset and os.path.exists(args.test_dataset):
        test_samples = load_jsonl(args.test_dataset, config.num_classes)
        test_loader = DataLoader(InvariantTraceDataset(test_samples, tokenizer, config.max_seq_len),
                                 batch_size=args.batch_size, shuffle=False)
        metrics["held_out_test"] = evaluate(model, test_loader, device, pad_id, config.num_classes)
        print(f"[+] Held-out test accuracy: {metrics['held_out_test']['accuracy'] * 100:.2f}% "
              f"on {metrics['held_out_test']['n']:,} samples")

    with open(os.path.join(args.checkpoint_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics (confusion matrix, per-class recall) saved to {os.path.join(args.checkpoint_dir, 'metrics.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ledger Invariant Model")
    parser.add_argument("--dataset", type=str, default="data/synthetic_traces_50k.jsonl")
    parser.add_argument("--test_dataset", type=str, default="data/traces_test_v01.jsonl",
                        help="Optional held-out JSONL evaluated once at the end.")
    parser.add_argument("--model_size", type=str, choices=["10m", "35m"], default="10m")
    parser.add_argument("--num_classes", type=int, choices=[2, 3], default=2,
                        help="2 = v0.1 ALLOW/BLOCK gatekeeper, 3 = v0.2 with ROLLBACK.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
