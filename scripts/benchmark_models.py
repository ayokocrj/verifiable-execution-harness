# -*- coding: utf-8 -*-
"""
CLI script to evaluate and benchmark verifier models.
Usage:
    python scripts/benchmark_models.py --samples 1000
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.eval_suite import BenchmarkSuite

def main():
    parser = argparse.ArgumentParser(description="Run Ledger-Invariant Benchmark Suite.")
    parser.add_argument("--samples", type=int, default=1_000, help="Number of evaluation samples.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained PyTorch model checkpoint (.pt).")
    parser.add_argument("--include_rollback", action="store_true", help="Evaluate v0.2 split with ROLLBACK.")
    args = parser.parse_args()

    target_desc = f"with model '{args.model_path}'" if args.model_path else "using Symbolic Oracle"
    print(f"[*] Running Ledger-Invariant Benchmark on {args.samples:,} test samples ({target_desc})...")
    suite = BenchmarkSuite(
        test_samples=args.samples,
        model_path=args.model_path,
        include_rollback=args.include_rollback,
    )
    results = suite.run_evaluation()

    print("\n" + "="*80)
    print(" " * 20 + "LEDGER-INVARIANT BENCHMARK RESULTS")
    print("="*80)
    print(suite.print_markdown_table(results))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
