# -*- coding: utf-8 -*-
"""
Ledger-Invariant Benchmark Evaluation Suite (ledger-invariant-bench).

Every number reported here is measured on the generated test set. Nothing is simulated:
  - No baseline is allowed to read the ground-truth label.
  - Latencies are wall-clock measurements, never constants.
  - A zero-shot LLM baseline is only reported when a real callable is injected via
    `llm_zero_shot_fn`; otherwise that row is absent (not faked).
  - The "Ledger-Invariant" row is only labelled as the learned model when neural weights
    are actually loaded. Without weights the engine falls back to the symbolic rule oracle,
    and the row is labelled as such (it is the upper bound the model is trained to approximate).
"""

import time
import random
from collections import Counter
from typing import Callable, Dict, Any, List, Optional, Tuple
from synthetic_data.generator import SyntheticTraceGenerator, TraceSample
from models.decision_engine import DecisionEngine

LABELS = ["ALLOW", "BLOCK", "ROLLBACK"]


class BaselineEvaluator:
    @staticmethod
    def random_baseline(sample: TraceSample, rng: random.Random, labels: List[str] = LABELS) -> str:
        return rng.choice(labels)

    @staticmethod
    def fit_majority(train: List[TraceSample]) -> str:
        return Counter(s.label for s in train).most_common(1)[0][0]

    @staticmethod
    def fit_action_name_lookup(train: List[TraceSample], fallback: str) -> Dict[str, str]:
        """
        Leakage probe: predicts the label from the action name alone.
        On a sound dataset this must not beat the majority class by more than a few points.
        """
        per_action: Dict[str, Counter] = {}
        for s in train:
            per_action.setdefault(s.action_name, Counter())[s.label] += 1
        return {a: c.most_common(1)[0][0] for a, c in per_action.items()}

    @staticmethod
    def heuristic_rule_baseline(sample: TraceSample) -> str:
        # Static threshold heuristic without contextual state tracking
        params = sample.action_params
        if "amount" in params and params["amount"] > 3_000_000.0:
            return "BLOCK"
        if "voltage" in params and (params["voltage"] < 95.0 or params["voltage"] > 125.0):
            return "BLOCK"
        if "current" in params and params["current"] > 750.0:
            return "BLOCK"
        return "ALLOW"


class BenchmarkSuite:
    def __init__(
        self,
        test_samples: int = 1_000,
        model_path: Optional[str] = None,
        llm_zero_shot_fn: Optional[Callable[[TraceSample], str]] = None,
        seed: int = 1337,
        include_rollback: bool = False,
    ):
        """include_rollback=False evaluates the v0.1 binary split; True evaluates the v0.2 split."""
        gen = SyntheticTraceGenerator(seed=seed)
        self.samples = gen.generate_dataset(
            total_samples=test_samples, adversarial_ratio=0.4, include_rollback=include_rollback
        )
        self.labels = LABELS if include_rollback else LABELS[:2]
        # Fitted baselines are trained on a disjoint split (different seed), never on the test set.
        fit_gen = SyntheticTraceGenerator(seed=seed + 1)
        fit_split = fit_gen.generate_dataset(total_samples=5_000, adversarial_ratio=0.4, include_rollback=include_rollback)
        self.majority_label = BaselineEvaluator.fit_majority(fit_split)
        self.action_lookup = BaselineEvaluator.fit_action_name_lookup(fit_split, self.majority_label)
        self.engine = DecisionEngine(model_path=model_path)
        self.llm_zero_shot_fn = llm_zero_shot_fn
        self.rng = random.Random(seed)

    def _baselines(self) -> List[Tuple[str, Callable[[TraceSample], str], bool]]:
        """(display name, predictor, is_learned_model)"""
        baselines: List[Tuple[str, Callable[[TraceSample], str], bool]] = [
            ("Random Baseline", lambda s: BaselineEvaluator.random_baseline(s, self.rng, self.labels), False),
            ("Majority Class", lambda s: self.majority_label, False),
            ("Action-Name Lookup (leakage probe)", lambda s: self.action_lookup.get(s.action_name, self.majority_label), False),
            ("Heuristic Rules", BaselineEvaluator.heuristic_rule_baseline, False),
        ]
        if self.llm_zero_shot_fn is not None:
            baselines.append(("LLM Zero-Shot (measured)", self.llm_zero_shot_fn, False))

        engine_fn = lambda s: self.engine.predict(s.state, s.action_name, s.action_params, s.invariants).decision
        if self.engine.has_weights:
            baselines.append(("Ledger-Invariant (Ours, neural)", engine_fn, True))
        else:
            baselines.append(("Symbolic Oracle (rule upper bound, no weights loaded)", engine_fn, False))
        return baselines

    def run_evaluation(self) -> Dict[str, Dict[str, Any]]:
        results = {}

        for name, eval_fn, is_learned in self._baselines():
            correct = 0
            adversarial_total = 0
            adversarial_caught = 0
            t0 = time.perf_counter()

            for s in self.samples:
                pred = eval_fn(s)
                if pred == s.label:
                    correct += 1
                if s.label in ["BLOCK", "ROLLBACK"]:
                    adversarial_total += 1
                    if pred == s.label:
                        adversarial_caught += 1

            total_time_ms = (time.perf_counter() - t0) * 1000.0
            avg_latency = total_time_ms / max(1, len(self.samples))

            acc = correct / len(self.samples)
            adv_recall = adversarial_caught / max(1, adversarial_total)

            results[name] = {
                "accuracy": round(acc * 100.0, 2),
                "adversarial_recall": round(adv_recall * 100.0, 2),
                "latency_ms": round(avg_latency, 3),
                "learned_model": is_learned,
                "n_samples": len(self.samples),
            }

        return results

    def print_markdown_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        header = "| Verifier | Invariant Accuracy | Adversarial Recall | Latency (ms, measured) | Learned model |\n"
        divider = "| :--- | :---: | :---: | :---: | :---: |\n"
        rows = []
        for model, m in results.items():
            bold = "**" if m["learned_model"] else ""
            learned = "yes" if m["learned_model"] else "no"
            row = f"| {bold}{model}{bold} | {bold}{m['accuracy']}%{bold} | {bold}{m['adversarial_recall']}%{bold} | {m['latency_ms']} | {learned} |"
            rows.append(row)
        return header + divider + "\n".join(rows)
