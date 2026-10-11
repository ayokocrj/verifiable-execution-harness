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

if __name__ == "__main__":
    unittest.main()
