# -*- coding: utf-8 -*-
import unittest
from models.tokenizer import InvariantTokenizer
from models.architecture import LedgerInvariantConfig
from models.decision_engine import DecisionEngine

class TestModelComponents(unittest.TestCase):

    def setUp(self):
        self.tokenizer = InvariantTokenizer()
        self.config_10m = LedgerInvariantConfig.get_10m_config()
        self.config_35m = LedgerInvariantConfig.get_35m_config()

    def test_parameter_counts(self):
        # Verify calculated parameter sizes correspond to targets
        params_10m = self.config_10m.count_parameters()
        params_35m = self.config_35m.count_parameters()
        self.assertTrue(5_000_000 < params_10m < 20_000_000, f"10M config params: {params_10m}")
        self.assertTrue(25_000_000 < params_35m < 50_000_000, f"35M config params: {params_35m}")

    def test_tokenizer_encoding(self):
        text = "[STATE] balance:5000000.0 [ACTION] TRANSFER amount:100000.0 [CRITERIA] ZERO_SUM"
        token_ids = self.tokenizer.encode(text, max_length=64)
        self.assertGreater(len(token_ids), 5)
        self.assertEqual(token_ids[0], self.tokenizer.token_to_id["[BOS]"])
        self.assertEqual(token_ids[-1], self.tokenizer.token_to_id["[EOS]"])

    def test_decision_engine_inference(self):
        engine = DecisionEngine()
        state = {"operating_reserve_usd": 5_000_000.0, "payroll_clearing_usd": 1_000_000.0}
        
        # Valid transfer -> ALLOW
        res_valid = engine.predict(
            state, "SETTLE_PAYROLL",
            {"source_account": "operating_reserve_usd", "dest_account": "payroll_clearing_usd", "amount": 500_000.0},
            ["TREASURY-ZERO-SUM-CONSERVATION", "MIN-OPERATING-LIQUIDITY-COVENANT"]
        )
        self.assertEqual(res_valid.decision, "ALLOW")
        self.assertLess(res_valid.latency_ms, 20.0) # sub-20ms latency

        # Excessive drain below $1M -> BLOCK
        res_invalid = engine.predict(
            state, "EXCESSIVE_ESCROW_DRAIN",
            {"source_account": "operating_reserve_usd", "dest_account": "vendor_escrow_usd", "amount": 4_500_000.0},
            ["MIN-OPERATING-LIQUIDITY-COVENANT"]
        )
        self.assertEqual(res_invalid.decision, "BLOCK")

if __name__ == "__main__":
    unittest.main()
