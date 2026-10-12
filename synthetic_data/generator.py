# -*- coding: utf-8 -*-
"""
Synthetic Trace Generator:
Generates deterministic execution traces with mathematical ground truth.

Anti-leakage design (v0.1 audit, 2026-10-12):
  - Every action name occurs with BOTH labels. The label is computed by
    `synthetic_data.rules.oracle_decision` from (state, params) only, so a model cannot
    learn a shortcut on the action name.
  - A share of samples is drawn within a few percent of each threshold (hard cases), so
    the task really is "compare numbers to a covenant", not "spot an outlier".
"""

import json
import random
from typing import Dict, Any, List, Optional

from synthetic_data.rules import (
    oracle_decision, LABEL_TO_CLASS,
    MIN_OPERATING_LIQUIDITY_USD, VOLTAGE_MIN_KV, VOLTAGE_MAX_KV, CURRENT_MAX_A,
)


class TraceSample:
    def __init__(
        self,
        domain: str,
        state: Dict[str, Any],
        action_name: str,
        action_params: Dict[str, Any],
        invariants: List[str],
        label: str,
        class_id: int,
        breach_reason: Optional[str] = None
    ):
        self.domain = domain
        self.state = state
        self.action_name = action_name
        self.action_params = action_params
        self.invariants = invariants
        self.label = label                    # "ALLOW", "BLOCK", "ROLLBACK"
        self.class_id = class_id              # 0, 1, 2
        self.breach_reason = breach_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "state": self.state,
            "action_name": self.action_name,
            "action_params": self.action_params,
            "invariants": self.invariants,
            "label": self.label,
            "class_id": self.class_id,
            "breach_reason": self.breach_reason
        }


class SyntheticTraceGenerator:
    """
    Deterministic trace generation across 2 domains:
    1. Financial Treasury Operations (liquidity covenant, sign constraint, double-entry conservation)
    2. SCADA Industrial Substation (voltage envelope, overcurrent protection)
    """

    V01_LABELS = ["ALLOW", "BLOCK"]
    V02_LABELS = ["ALLOW", "BLOCK", "ROLLBACK"]

    TREASURY_INVARIANTS = ["TREASURY-ZERO-SUM-CONSERVATION", "MIN-OPERATING-LIQUIDITY-COVENANT"]
    SCADA_INVARIANTS = ["GRID-VOLTAGE-STABILITY", "OVERCURRENT-PROTECTION"]

    # Share of samples drawn within NEAR_BAND of the decision threshold.
    HARD_CASE_RATIO = 0.30
    NEAR_BAND = 0.02

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers
    def _near(self, threshold: float, below: bool, floor: float = 0.01) -> float:
        """A value just below or just above `threshold`, within NEAR_BAND (relative) of it."""
        band = max(abs(threshold) * self.NEAR_BAND, 1.0)
        delta = self.rng.uniform(floor, band)
        return threshold - delta if below else threshold + delta

    def _checksum(self) -> str:
        return f"{self.rng.randrange(100_000, 999_999)}"

    def _finish(self, domain: str, state: Dict[str, Any], action: str, params: Dict[str, Any],
                invariants: List[str]) -> TraceSample:
        label, reason = oracle_decision(state, action, params)
        return TraceSample(domain, state, action, params, invariants, label, LABEL_TO_CLASS[label], reason)

    # ------------------------------------------------------------------ treasury
    def generate_treasury_trace(self, is_adversarial: bool = False, include_rollback: bool = True) -> TraceSample:
        op_reserve = round(self.rng.uniform(1_200_000.0, 10_000_000.0), 2)
        payroll = round(self.rng.uniform(500_000.0, 2_000_000.0), 2)
        escrow = round(self.rng.uniform(200_000.0, 1_500_000.0), 2)
        state: Dict[str, Any] = {
            "operating_reserve_usd": op_reserve,
            "payroll_clearing_usd": payroll,
            "vendor_escrow_usd": escrow,
        }
        if include_rollback:
            state["ledger_checksum"] = self._checksum()

        kind = self.rng.choice(["transfer", "transfer", "ledger_entry"])

        if kind == "ledger_entry":
            # POST_LEDGER_ENTRY appears with both labels: legs equal -> ALLOW, legs differ -> BLOCK.
            debit = round(self.rng.uniform(10_000.0, 2_000_000.0), 2)
            if not is_adversarial:
                credit = debit
            else:
                hard = self.rng.random() < self.HARD_CASE_RATIO
                gap = self.rng.uniform(0.01, 5.0) if hard else self.rng.uniform(5.0, 0.1 * debit)
                credit = round(debit + self.rng.choice([-1, 1]) * gap, 2)
                if credit == debit:
                    credit = round(debit + 0.01, 2)
            params = {"debit_account": "vendor_escrow_usd", "credit_account": "operating_reserve_usd",
                      "debit": debit, "credit": credit}
            return self._finish("treasury", state, "POST_LEDGER_ENTRY", params, self.TREASURY_INVARIANTS)

        # Transfers out of the operating reserve: same action names on both sides.
        action = self.rng.choice(["SETTLE_PAYROLL", "FUND_VENDOR_ESCROW"])
        dest = "payroll_clearing_usd" if action == "SETTLE_PAYROLL" else "vendor_escrow_usd"
        max_allowed = round(op_reserve - MIN_OPERATING_LIQUIDITY_USD, 2)   # largest amount that keeps the covenant
        hard = self.rng.random() < self.HARD_CASE_RATIO

        if not is_adversarial:
            amount = self._near(max_allowed, below=True) if hard else self.rng.uniform(1_000.0, max_allowed)
            amount = round(max(amount, 0.01), 2)
            params = {"source_account": "operating_reserve_usd", "dest_account": dest, "amount": amount}
            if include_rollback:
                params["expected_checksum"] = state["ledger_checksum"]
            return self._finish("treasury", state, action, params, self.TREASURY_INVARIANTS)

        attacks = ["covenant_breach", "covenant_breach", "negative_amount"]
        if include_rollback:
            attacks.append("state_divergence")
        attack = self.rng.choice(attacks)

        if attack == "covenant_breach":
            amount = self._near(max_allowed, below=False) if hard else self.rng.uniform(max_allowed + 1.0, op_reserve)
            amount = round(amount, 2)
        elif attack == "negative_amount":
            amount = -round(self.rng.uniform(0.01, 5.0) if hard else self.rng.uniform(1_000.0, 500_000.0), 2)
        else:  # state_divergence: amounts are valid, only the checksum disagrees
            amount = round(self.rng.uniform(1_000.0, max_allowed), 2)

        params = {"source_account": "operating_reserve_usd", "dest_account": dest, "amount": amount}
        if include_rollback:
            if attack == "state_divergence":
                other = self._checksum()
                while other == state["ledger_checksum"]:
                    other = self._checksum()
                params["expected_checksum"] = other
            else:
                params["expected_checksum"] = state["ledger_checksum"]
        return self._finish("treasury", state, action, params, self.TREASURY_INVARIANTS)

    # ------------------------------------------------------------------ scada
    def generate_scada_trace(self, is_adversarial: bool = False) -> TraceSample:
        voltage = round(self.rng.uniform(105.0, 120.0), 1)
        current = round(self.rng.uniform(200.0, 500.0), 1)
        state = {"bus_voltage_kv": voltage, "feeder_current_a": current, "breaker_closed": True}
        action = self.rng.choice(["ADJUST_TAP_CHANGER", "DISPATCH_LOAD"])
        hard = self.rng.random() < self.HARD_CASE_RATIO

        if action == "ADJUST_TAP_CHANGER":
            if not is_adversarial:
                if hard:
                    edge = self.rng.choice([VOLTAGE_MIN_KV, VOLTAGE_MAX_KV])
                    target = self._near(edge, below=(edge == VOLTAGE_MAX_KV), floor=0.1)
                else:
                    target = self.rng.uniform(VOLTAGE_MIN_KV + 1.0, VOLTAGE_MAX_KV - 1.0)
            else:
                if hard:
                    edge = self.rng.choice([VOLTAGE_MIN_KV, VOLTAGE_MAX_KV])
                    target = self._near(edge, below=(edge == VOLTAGE_MIN_KV), floor=0.1)
                else:
                    target = self.rng.choice([self.rng.uniform(60.0, VOLTAGE_MIN_KV - 1.0),
                                              self.rng.uniform(VOLTAGE_MAX_KV + 1.0, 160.0)])
            params = {"voltage": round(target, 1)}
        else:  # DISPATCH_LOAD
            if not is_adversarial:
                target = self._near(CURRENT_MAX_A, below=True, floor=0.1) if hard else self.rng.uniform(50.0, CURRENT_MAX_A - 1.0)
            else:
                if hard:
                    target = self._near(CURRENT_MAX_A, below=False, floor=0.1)
                else:
                    target = self.rng.choice([self.rng.uniform(CURRENT_MAX_A + 1.0, 1_400.0),
                                              -self.rng.uniform(1.0, 200.0)])
            params = {"current": round(target, 1)}

        return self._finish("scada", state, action, params, self.SCADA_INVARIANTS)

    # ------------------------------------------------------------------ dataset
    def generate_dataset(
        self,
        total_samples: int = 50_000,
        adversarial_ratio: float = 0.5,
        include_rollback: bool = False,
    ) -> List[TraceSample]:
        """
        include_rollback=False (default) produces the v0.1 binary split (ALLOW / BLOCK).
        include_rollback=True adds the post-action ROLLBACK class for the v0.2 split.
        adversarial_ratio is the share of violating samples; 0.5 gives balanced labels.
        """
        samples: List[TraceSample] = []
        for _ in range(total_samples):
            is_adv = (self.rng.random() < adversarial_ratio)
            domain = self.rng.choice(["treasury", "treasury", "scada"])  # 66% treasury, 33% scada
            if domain == "treasury":
                samples.append(self.generate_treasury_trace(is_adversarial=is_adv, include_rollback=include_rollback))
            else:
                samples.append(self.generate_scada_trace(is_adversarial=is_adv))
        return samples

    def export_jsonl(self, samples: List[TraceSample], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")
