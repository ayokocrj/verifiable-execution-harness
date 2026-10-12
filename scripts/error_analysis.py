# -*- coding: utf-8 -*-
"""
Error analysis for a trained ledger-invariant checkpoint.

Buckets every held-out sample by how close its parameters sit to the invariant threshold,
then reports accuracy per bucket, per action, per domain, and the confusion matrix.
This is the measurement behind any "errors concentrate near the boundary" claim.

Usage (Colab, after training):
    python scripts/error_analysis.py --model_path checkpoints/ledger_invariant_10m_final.pt \
        --dataset data/traces_test_v01.jsonl --out checkpoints/error_analysis.json
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.decision_engine import DecisionEngine
from synthetic_data.rules import MIN_OPERATING_LIQUIDITY_USD, VOLTAGE_MIN_KV, VOLTAGE_MAX_KV, CURRENT_MAX_A

BUCKETS = [(0.0, 0.01, "<1%"), (0.01, 0.02, "1-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.20, "5-20%"), (0.20, float("inf"), ">20%")]


def relative_margin(state: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Relative distance to the nearest decision threshold, and the rule family it belongs to."""
    if "debit" in params and "credit" in params:
        debit = float(params["debit"])
        return abs(debit - float(params["credit"])) / max(debit, 1.0), "double_entry"
    if "amount" in params:
        amount = float(params["amount"])
        if amount <= 0:
            return None, "sign_constraint"
        src = params.get("source_account")
        if src in state and "operating_reserve" in src:
            headroom = float(state[src]) - MIN_OPERATING_LIQUIDITY_USD
            return abs(headroom - amount) / max(headroom, 1.0), "liquidity_covenant"
        return None, "transfer_other"
    if "voltage" in params:
        v = float(params["voltage"])
        edge = VOLTAGE_MIN_KV if abs(v - VOLTAGE_MIN_KV) < abs(v - VOLTAGE_MAX_KV) else VOLTAGE_MAX_KV
        return abs(v - edge) / edge, "voltage_envelope"
    if "current" in params:
        c = float(params["current"])
        if c < 0:
            return None, "sign_constraint"
        return abs(c - CURRENT_MAX_A) / CURRENT_MAX_A, "overcurrent"
    return None, "other"


def bucket_of(margin: Optional[float]) -> str:
    if margin is None:
        return "sign/other"
    for lo, hi, name in BUCKETS:
        if lo <= margin < hi:
            return name
    return ">20%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="data/traces_test_v01.jsonl")
    parser.add_argument("--out", default="checkpoints/error_analysis.json")
    args = parser.parse_args()

    engine = DecisionEngine(model_path=args.model_path)
    if not engine.has_weights:
        print("[!] Weights could not be loaded; refusing to analyse the symbolic oracle instead.")
        sys.exit(1)

    with open(args.dataset, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    labels = sorted({s["label"] for s in samples})
    confusion = {t: {p: 0 for p in labels} for t in labels}
    by_bucket, by_action, by_domain, by_rule = (defaultdict(lambda: [0, 0]) for _ in range(4))
    errors = []

    for s in samples:
        pred = engine.predict(s["state"], s["action_name"], s["action_params"], s["invariants"]).decision
        ok = pred == s["label"]
        confusion[s["label"]][pred] = confusion[s["label"]].get(pred, 0) + 1
        margin, rule = relative_margin(s["state"], s["action_params"])
        for table, key in ((by_bucket, bucket_of(margin)), (by_action, s["action_name"]),
                           (by_domain, s["domain"]), (by_rule, rule)):
            table[key][1] += 1
            table[key][0] += int(ok)
        if not ok:
            errors.append({"label": s["label"], "pred": pred, "action": s["action_name"], "rule": rule,
                           "relative_margin": margin, "state": s["state"], "params": s["action_params"]})

    def table(d):
        return {k: {"accuracy": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(d.items(), key=lambda kv: kv[0])}

    report = {
        "n": len(samples),
        "accuracy": round(sum(v[0] for v in by_bucket.values()) / len(samples), 4),
        "confusion_matrix": confusion,
        "accuracy_by_margin_bucket": table(by_bucket),
        "accuracy_by_rule": table(by_rule),
        "accuracy_by_action": table(by_action),
        "accuracy_by_domain": table(by_domain),
        "errors": errors,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAccuracy: {report['accuracy'] * 100:.2f}% on {len(samples)} samples\n")
    for title, key in (("Distance to threshold", "accuracy_by_margin_bucket"), ("Rule family", "accuracy_by_rule"),
                       ("Action", "accuracy_by_action"), ("Domain", "accuracy_by_domain")):
        print(f"| {title} | Accuracy | n |\n| :--- | :---: | :---: |")
        for k, v in report[key].items():
            print(f"| {k} | {v['accuracy'] * 100:.1f}% | {v['n']} |")
        print()
    print("Confusion (rows = truth, cols = prediction):")
    for t in labels:
        print(f"  {t:8s} " + "  ".join(f"{p}={confusion[t][p]}" for p in labels))
    print(f"\n[+] Full report with every error written to {args.out}")


if __name__ == "__main__":
    main()
