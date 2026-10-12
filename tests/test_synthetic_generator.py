# -*- coding: utf-8 -*-
import unittest
from synthetic_data.generator import SyntheticTraceGenerator

class TestSyntheticGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = SyntheticTraceGenerator(seed=123)

    def test_trace_generation_balance(self):
        samples = self.generator.generate_dataset(total_samples=100, adversarial_ratio=0.3)
        self.assertEqual(len(samples), 100)

        allow_count = sum(1 for s in samples if s.label == "ALLOW")
        block_or_rollback = sum(1 for s in samples if s.label in ["BLOCK", "ROLLBACK"])

        self.assertGreater(allow_count, 50)
        self.assertGreater(block_or_rollback, 15)

    def test_treasury_adversarial_reasons(self):
        sample = self.generator.generate_treasury_trace(is_adversarial=True)
        self.assertIn(sample.label, ["BLOCK", "ROLLBACK"])
        self.assertIsNotNone(sample.breach_reason)

    def test_v01_split_is_binary(self):
        samples = self.generator.generate_dataset(total_samples=2000, adversarial_ratio=0.5)
        labels = {s.label for s in samples}
        self.assertEqual(labels, {"ALLOW", "BLOCK"})
        self.assertTrue(all(s.class_id in (0, 1) for s in samples))

    def test_v02_split_contains_rollback(self):
        samples = self.generator.generate_dataset(total_samples=2000, adversarial_ratio=0.5, include_rollback=True)
        self.assertIn("ROLLBACK", {s.label for s in samples})
        # ROLLBACK must be visible in the input: checksum carried by the action differs from the ledger's
        for s in samples:
            if s.label == "ROLLBACK":
                self.assertNotEqual(s.action_params["expected_checksum"], s.state["ledger_checksum"])

    def test_symbolic_oracle_is_exact_on_v02_split(self):
        from models.decision_engine import DecisionEngine
        engine = DecisionEngine()
        samples = self.generator.generate_dataset(total_samples=1000, adversarial_ratio=0.5, include_rollback=True)
        mismatches = sum(
            engine.predict(s.state, s.action_name, s.action_params, s.invariants).decision != s.label
            for s in samples
        )
        self.assertEqual(mismatches, 0)

    def test_action_name_is_not_predictive(self):
        """Anti-leakage: every action name must occur with both labels, and a lookup on the
        action name must not beat the majority class by more than 5 points."""
        from collections import Counter
        from benchmarks.eval_suite import BaselineEvaluator
        fit = SyntheticTraceGenerator(seed=11).generate_dataset(total_samples=5000, adversarial_ratio=0.5)
        test = SyntheticTraceGenerator(seed=12).generate_dataset(total_samples=2000, adversarial_ratio=0.5)
        per_action = {}
        for s in fit:
            per_action.setdefault(s.action_name, set()).add(s.label)
        for action, labels in per_action.items():
            self.assertEqual(labels, {"ALLOW", "BLOCK"}, f"{action} only ever has labels {labels}")
        majority = BaselineEvaluator.fit_majority(fit)
        lookup = BaselineEvaluator.fit_action_name_lookup(fit, majority)
        acc_majority = sum(s.label == majority for s in test) / len(test)
        acc_lookup = sum(lookup.get(s.action_name, majority) == s.label for s in test) / len(test)
        self.assertLessEqual(acc_lookup, acc_majority + 0.05,
                             f"action-name lookup {acc_lookup:.3f} vs majority {acc_majority:.3f}: label leaks through the action name")

    def test_hard_cases_exist_near_thresholds(self):
        """Some ALLOW and BLOCK samples must sit within 2% of the liquidity covenant."""
        from synthetic_data.rules import MIN_OPERATING_LIQUIDITY_USD
        samples = self.generator.generate_dataset(total_samples=3000, adversarial_ratio=0.5)
        near_allow = near_block = 0
        for s in samples:
            p = s.action_params
            if s.domain == "treasury" and "amount" in p and p["amount"] > 0:
                margin = s.state["operating_reserve_usd"] - p["amount"] - MIN_OPERATING_LIQUIDITY_USD
                if abs(margin) <= 0.02 * (s.state["operating_reserve_usd"] - MIN_OPERATING_LIQUIDITY_USD):
                    if s.label == "ALLOW":
                        near_allow += 1
                    else:
                        near_block += 1
        self.assertGreater(near_allow, 50)
        self.assertGreater(near_block, 50)

    def test_symbolic_oracle_is_exact_on_v01_split(self):
        """The generator and the rule oracle must agree 100% on the binary split."""
        from models.decision_engine import DecisionEngine
        engine = DecisionEngine()
        samples = self.generator.generate_dataset(total_samples=1000, adversarial_ratio=0.5)
        mismatches = [
            s for s in samples
            if engine.predict(s.state, s.action_name, s.action_params, s.invariants).decision != s.label
        ]
        self.assertEqual(len(mismatches), 0, f"first mismatch: {mismatches[0].to_dict() if mismatches else None}")

if __name__ == "__main__":
    unittest.main()
