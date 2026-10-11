# -*- coding: utf-8 -*-
"""
PyTorch Training Script for Ledger-Invariant-35M Decision Model.
Usage:
    python scripts/train.py --dataset data/synthetic_traces_50k.jsonl --epochs 5 --batch_size 64
"""

import os
import sys
import json
import argparse
from typing import List, Tuple

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

if TORCH_AVAILABLE:
    class InvariantTraceDataset(Dataset):
        def __init__(self, data_path: str, tokenizer: InvariantTokenizer, max_len: int = 128):
            self.samples = []
            self.tokenizer = tokenizer
            self.max_len = max_len

            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.samples.append(json.loads(line))

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

def train(args):
    if not TORCH_AVAILABLE:
        print("[!] PyTorch is not installed in the local environment. Please run on GPU (Colab/RunPod).")
        return

    tokenizer = InvariantTokenizer()
    config = LedgerInvariantConfig.get_35m_config() if args.model_size == "35m" else LedgerInvariantConfig.get_10m_config()

    print(f"[*] Initializing model {args.model_size.upper()} ({config.count_parameters():,} parameters)...")
    model = LedgerInvariantDecisionTransformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[*] Training on device: {device}")

    # Dataset
    if not os.path.exists(args.dataset):
        print(f"[!] Dataset {args.dataset} not found. Generate it first using scripts/generate_dataset.py.")
        return

    dataset = InvariantTraceDataset(args.dataset, tokenizer, max_len=config.max_seq_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"[*] Starting training loop for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = (correct / total) * 100.0 if total > 0 else 0.0
        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.2f}%")

    out_weights = os.path.join(args.checkpoint_dir, f"ledger_invariant_{args.model_size}_final.pt")
    torch.save(model.state_dict(), out_weights)
    print(f"[+] Model weights saved to {out_weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ledger Invariant Model")
    parser.add_argument("--dataset", type=str, default="data/synthetic_traces_50k.jsonl")
    parser.add_argument("--model_size", type=str, choices=["10m", "35m"], default="35m")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
