# -*- coding: utf-8 -*-
"""
CLI script to generate synthetic trace datasets for pretraining and benchmarking.
Usage:
    python scripts/generate_dataset.py --samples 50000 --out data/traces_50k.jsonl
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthetic_data.generator import SyntheticTraceGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic invariant traces.")
    parser.add_argument("--samples", type=int, default=50_000, help="Total samples to generate.")
    parser.add_argument("--out", type=str, default="data/synthetic_traces_50k.jsonl", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--adversarial_ratio", type=float, default=0.35)
    parser.add_argument("--include_rollback", action="store_true",
                        help="Add the post-action ROLLBACK class (v0.2 split). Off by default: v0.1 is ALLOW/BLOCK.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[*] Initializing SyntheticTraceGenerator with seed={args.seed}...")
    gen = SyntheticTraceGenerator(seed=args.seed)

    split = "v0.2 (ALLOW/BLOCK/ROLLBACK)" if args.include_rollback else "v0.1 (ALLOW/BLOCK)"
    print(f"[*] Generating {args.samples:,} deterministic execution traces, split {split}...")
    samples = gen.generate_dataset(
        total_samples=args.samples,
        adversarial_ratio=args.adversarial_ratio,
        include_rollback=args.include_rollback,
    )

    print(f"[*] Writing dataset to {args.out}...")
    gen.export_jsonl(samples, args.out)

    counts = Counter(s.label for s in samples)
    domains = Counter(s.domain for s in samples)
    print(f"[+] Done! Exported {len(samples):,} traces. Labels: {dict(counts)}. Domains: {dict(domains)}.")

if __name__ == "__main__":
    main()
