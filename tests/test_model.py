# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import unittest

from models.tokenizer import InvariantTokenizer, SPECIAL_TOKENS
from models.architecture import LedgerInvariantConfig
from models.decision_engine import DecisionEngine

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_TEXT = (
    "[STATE] operating_reserve_usd:8190853.5 payroll_clearing_usd:867337.78 "
    "[ACTION] SETTLE_PAYROLL amount:598597.29 source_account:operating_reserve_usd "
    "[CRITERIA] TREASURY-ZERO-SUM-CONSERVATION MIN-OPERATING-LIQUIDITY-COVENANT"
)


class TestModelComponents(unittest.TestCase):

    def setUp(self):
        self.tokenizer = InvariantTokenizer()
        self.config_10m = LedgerInvariantConfig.get_10m_config()
        self.config_35m = LedgerInvariantConfig.get_35m_config()

    def test_parameter_counts(self):
        params_10m = self.config_10m.count_parameters()
        params_35m = self.config_35m.count_parameters()
        self.assertTrue(5_000_000 < params_10m < 20_000_000, f"10M config params: {params_10m}")
        self.assertTrue(25_000_000 < params_35m < 50_000_000, f"35M config params: {params_35m}")

    def test_v01_config_is_binary_and_round_trips(self):
        import tempfile, os
        self.assertEqual(self.config_10m.num_classes, 2)
        self.assertEqual(self.config_35m.num_classes, 3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            self.config_10m.save(path)
            loaded = LedgerInvariantConfig.load(path)
        self.assertEqual(loaded.to_dict(), self.config_10m.to_dict())

    def test_formatted_traces_fit_in_context_window(self):
        """Digit-level tokens lengthen traces; every generated trace must fit max_seq_len."""
        from synthetic_data.generator import SyntheticTraceGenerator
        gen = SyntheticTraceGenerator(seed=7)
        samples = gen.generate_dataset(total_samples=2000, adversarial_ratio=0.5, include_rollback=True)
        budget = self.config_10m.max_seq_len - 2  # room for [BOS] and [EOS]
        longest = max(
            len(self.tokenizer._tokenize_text(
                self.tokenizer.format_input(s.state, s.action_name, s.action_params, s.invariants)))
            for s in samples
        )
        self.assertLessEqual(longest, budget, f"longest trace = {longest} tokens > budget {budget}")

    def test_tokenizer_encoding(self):
        text = "[STATE] balance:5000000.0 [ACTION] TRANSFER amount:100000.0 [CRITERIA] ZERO_SUM"
        token_ids = self.tokenizer.encode(text, max_length=64)
        self.assertGreater(len(token_ids), 5)
        self.assertEqual(token_ids[0], self.tokenizer.token_to_id["[BOS]"])
        self.assertEqual(token_ids[-1], self.tokenizer.token_to_id["[EOS]"])

    def test_decision_engine_inference(self):
        engine = DecisionEngine()
        state = {"operating_reserve_usd": 5_000_000.0, "payroll_clearing_usd": 1_000_000.0}

        res_valid = engine.predict(
            state, "SETTLE_PAYROLL",
            {"source_account": "operating_reserve_usd", "dest_account": "payroll_clearing_usd", "amount": 500_000.0},
            ["TREASURY-ZERO-SUM-CONSERVATION", "MIN-OPERATING-LIQUIDITY-COVENANT"]
        )
        self.assertEqual(res_valid.decision, "ALLOW")
        self.assertLess(res_valid.latency_ms, 20.0)

        res_invalid = engine.predict(
            state, "EXCESSIVE_ESCROW_DRAIN",
            {"source_account": "operating_reserve_usd", "dest_account": "vendor_escrow_usd", "amount": 4_500_000.0},
            ["MIN-OPERATING-LIQUIDITY-COVENANT"]
        )
        self.assertEqual(res_invalid.decision, "BLOCK")


class TestTokenizerRegressions(unittest.TestCase):
    """Regression tests for the three v0.1 blockers: determinism, digit tokens, special markers."""

    def setUp(self):
        self.tokenizer = InvariantTokenizer()

    def test_special_tokens_keep_reserved_ids(self):
        for idx, tok in enumerate(SPECIAL_TOKENS):
            self.assertEqual(self.tokenizer.token_to_id[tok], idx)

    def test_markers_are_recognised_in_text(self):
        tokens = self.tokenizer._tokenize_text(SAMPLE_TEXT)
        self.assertIn("[STATE]", tokens)
        self.assertIn("[ACTION]", tokens)
        self.assertIn("[CRITERIA]", tokens)
        # Markers must never be shredded into "[", "STATE", "]"
        self.assertNotIn("STATE", tokens)
        ids = self.tokenizer.encode(SAMPLE_TEXT, max_length=256)
        self.assertIn(self.tokenizer.token_to_id["[STATE]"], ids)
        self.assertIn(self.tokenizer.token_to_id["[ACTION]"], ids)
        self.assertIn(self.tokenizer.token_to_id["[CRITERIA]"], ids)

    def test_numbers_split_digit_by_digit(self):
        tokens = self.tokenizer._tokenize_text("amount:598597.29")
        self.assertEqual(tokens, ["amount", ":", "5", "9", "8", "5", "9", "7", ".", "2", "9"])
        # Negative amounts keep an explicit sign token
        self.assertEqual(self.tokenizer._tokenize_text("x:-5.0"), ["x", ":", "-", "5", ".", "0"])
        # Identifiers that contain digits are not split
        self.assertEqual(self.tokenizer._tokenize_text("transformer_1"), ["transformer_1"])
        # Hyphenated invariant names stay whole
        self.assertEqual(self.tokenizer._tokenize_text("TREASURY-ZERO-SUM-CONSERVATION"),
                         ["TREASURY-ZERO-SUM-CONSERVATION"])
        # Every digit and the decimal point resolve to base-vocabulary ids, never fallback ids
        for d in "0123456789.":
            self.assertLess(self.tokenizer.token_to_id[d], self.tokenizer.base_size)

    def test_fallback_ids_never_collide_with_base_vocabulary(self):
        for tok in ["UNSEEN_TERM_X", "another-unseen", "ZZZ"]:
            fid = self.tokenizer.convert_token_to_id(tok)
            self.assertGreaterEqual(fid, self.tokenizer.base_size)
            self.assertLess(fid, self.tokenizer.vocab_size)

    def test_encoding_is_stable_across_instances(self):
        a = InvariantTokenizer().encode(SAMPLE_TEXT, max_length=256)
        b = InvariantTokenizer().encode(SAMPLE_TEXT, max_length=256)
        self.assertEqual(a, b)

    def test_encoding_is_stable_across_processes(self):
        """The original bug: hash() is salted per process, so ids changed between Colab and local runs."""
        script = (
            "import json; from models.tokenizer import InvariantTokenizer; "
            f"print(json.dumps(InvariantTokenizer().encode({SAMPLE_TEXT!r}, max_length=256)))"
        )
        runs = []
        for seed in ("1", "2", "random"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=REPO_ROOT)
            out = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
                                 capture_output=True, text=True, check=True)
            runs.append(json.loads(out.stdout))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])
        self.assertEqual(runs[0], self.tokenizer.encode(SAMPLE_TEXT, max_length=256))

    def test_eos_survives_truncation(self):
        ids = self.tokenizer.encode(SAMPLE_TEXT, max_length=16)
        self.assertEqual(len(ids), 16)
        self.assertEqual(ids[0], self.tokenizer.token_to_id["[BOS]"])
        self.assertEqual(ids[-1], self.tokenizer.token_to_id["[EOS]"])

    def test_pad_and_padding_mask(self):
        ids = self.tokenizer.encode("[STATE] a:1", max_length=32)
        padded = self.tokenizer.pad(ids, max_length=32)
        self.assertEqual(len(padded), 32)
        mask = self.tokenizer.padding_mask(padded)
        self.assertEqual(mask[:len(ids)], [False] * len(ids))
        self.assertTrue(all(mask[len(ids):]))


class TestBenchmarkHonesty(unittest.TestCase):
    """No baseline may read the label; no simulated LLM row; no hardcoded latency."""

    def test_no_simulated_llm_baseline(self):
        from benchmarks.eval_suite import BenchmarkSuite, BaselineEvaluator
        self.assertFalse(hasattr(BaselineEvaluator, "llm_simulation_baseline"))
        suite = BenchmarkSuite(test_samples=50)
        results = suite.run_evaluation()
        self.assertFalse(any("LLM" in k for k in results))
        # Without weights the engine row must not claim to be the learned model
        self.assertFalse(any(v["learned_model"] for v in results.values()))
        self.assertTrue(any("Oracle" in k for k in results))

    def test_llm_row_only_when_injected(self):
        from benchmarks.eval_suite import BenchmarkSuite
        suite = BenchmarkSuite(test_samples=20, llm_zero_shot_fn=lambda s: "ALLOW")
        results = suite.run_evaluation()
        self.assertTrue(any("LLM" in k for k in results))

    def test_latency_is_measured_not_constant(self):
        from benchmarks.eval_suite import BenchmarkSuite
        results = BenchmarkSuite(test_samples=20).run_evaluation()
        for v in results.values():
            self.assertNotEqual(v["latency_ms"], 1250.0)
            self.assertNotEqual(v["latency_ms"], 3.4)


if __name__ == "__main__":
    unittest.main()
