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
