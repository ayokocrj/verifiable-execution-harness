# -*- coding: utf-8 -*-
"""
CLI script to generate synthetic trace datasets for pretraining and benchmarking.
Usage:
    python scripts/generate_dataset.py --samples 50000 --out data/traces_50k.jsonl
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthetic_data.generator import SyntheticTraceGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic invariant traces.")
    parser.add_argument("--samples", type=int, default=50_000, help="Total samples to generate.")
    parser.add_argument("--out", type=str, default="data/synthetic_traces_50k.jsonl", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[*] Initializing SyntheticTraceGenerator with seed={args.seed}...")
    gen = SyntheticTraceGenerator(seed=args.seed)

    print(f"[*] Generating {args.samples:,} deterministic execution traces across 3 domains...")
    samples = gen.generate_dataset(total_samples=args.samples, adversarial_ratio=0.35)

    print(f"[*] Writing dataset to {args.out}...")
    gen.export_jsonl(samples, args.out)

    print(f"[+] Done! Successfully exported {len(samples):,} traces.")

if __name__ == "__main__":
    main()
