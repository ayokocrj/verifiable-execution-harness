# -*- coding: utf-8 -*-
"""
Synthetic Trace Generator:
Generates 50,000+ deterministic execution traces with mathematical ground truth.
Includes valid executions, edge cases, and precision adversarial fault injections.
"""

import json
import random
from typing import Dict, Any, List, Optional

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
    Deterministic trace generation harness across 3 domains:
    1. Financial Treasury Operations (Double-entry, liquidity covenant, escrow)
    2. SCADA Industrial Substation (Voltage bounds, surge current, breaker trips)
    3. Enterprise Billing & Reconciliations (Invoicing, tax balancing, refund limits)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_treasury_trace(self, is_adversarial: bool = False) -> TraceSample:
        # Initial treasury accounts
        op_reserve = round(self.rng.uniform(3_000_000.0, 10_000_000.0), 2)
        payroll = round(self.rng.uniform(500_000.0, 2_000_000.0), 2)
        escrow = round(self.rng.uniform(200_000.0, 1_500_000.0), 2)
        state = {
            "operating_reserve_usd": op_reserve,
            "payroll_clearing_usd": payroll,
            "vendor_escrow_usd": escrow
        }
        invariants = ["TREASURY-ZERO-SUM-CONSERVATION", "MIN-OPERATING-LIQUIDITY-COVENANT"]

        if not is_adversarial:
            # Valid Transfer: operating_reserve -> payroll_clearing
            max_transfer = op_reserve - 1_000_000.0 # Maintain $1M minimum reserve
            amount = round(self.rng.uniform(10_000.0, max(20_000.0, max_transfer * 0.8)), 2)
            return TraceSample(
                domain="treasury",
                state=state,
                action_name="SETTLE_PAYROLL",
                action_params={"source_account": "operating_reserve_usd", "dest_account": "payroll_clearing_usd", "amount": amount},
                invariants=invariants,
                label="ALLOW",
                class_id=0
            )
        else:
            # Precision adversarial corruption
            attack_type = self.rng.choice(["covenant_breach", "imbalanced_mutation", "negative_drain", "state_divergence"])

            if attack_type == "covenant_breach":
                # Drain reserve below $1M covenant
                excessive_amount = round(op_reserve - 500_000.0, 2)
                return TraceSample(
                    domain="treasury",
                    state=state,
                    action_name="EXCESSIVE_ESCROW_DRAIN",
                    action_params={"source_account": "operating_reserve_usd", "dest_account": "vendor_escrow_usd", "amount": excessive_amount},
                    invariants=invariants,
                    label="BLOCK",
                    class_id=1,
                    breach_reason="MIN-OPERATING-LIQUIDITY-COVENANT violated: reserve dropped below $1,000,000 threshold"
                )

            elif attack_type == "imbalanced_mutation":
                # Arbitrary single-sided credit (violation of double-entry conservation)
                mint_amount = round(self.rng.uniform(100_000.0, 2_000_000.0), 2)
                return TraceSample(
                    domain="treasury",
                    state=state,
                    action_name="INJECT_UNBALANCED_CREDIT",
                    action_params={"arbitrary_mutation": {"account": "operating_reserve_usd", "amount": mint_amount}},
                    invariants=invariants,
                    label="BLOCK",
                    class_id=1,
                    breach_reason="TREASURY-ZERO-SUM-CONSERVATION breached: unilateral asset expansion without matching debit"
                )

            elif attack_type == "negative_drain":
                # Negative amount injection
                return TraceSample(
                    domain="treasury",
                    state=state,
                    action_name="REVERSE_SWEEP",
                    action_params={"source_account": "operating_reserve_usd", "dest_account": "payroll_clearing_usd", "amount": -250_000.0},
                    invariants=invariants,
                    label="BLOCK",
                    class_id=1,
                    breach_reason="SIGN_CONSTRAINT violated: negative transaction amount attempted"
                )

            else:
                # Simulated post-action state divergence requiring rollback
                return TraceSample(
                    domain="treasury",
                    state=state,
                    action_name="CONCURRENT_SETTLEMENT",
                    action_params={"source_account": "operating_reserve_usd", "dest_account": "payroll_clearing_usd", "amount": 100_000.0},
                    invariants=invariants,
                    label="ROLLBACK",
                    class_id=2,
                    breach_reason="STATE_DIVERGENCE: detected lock collision or ledger checksum mismatch"
                )

    def generate_scada_trace(self, is_adversarial: bool = False) -> TraceSample:
        voltage = round(self.rng.uniform(105.0, 120.0), 1)
        current = round(self.rng.uniform(200.0, 500.0), 1)
        state = {"bus_voltage_kv": voltage, "feeder_current_a": current, "breaker_closed": True}
        invariants = ["GRID-VOLTAGE-STABILITY", "OVERCURRENT-PROTECTION"]

        if not is_adversarial:
            delta_v = round(self.rng.uniform(-3.0, 3.0), 1)
            target_v = round(voltage + delta_v, 1)
            return TraceSample(
                domain="scada",
                state=state,
                action_name="ADJUST_TAP_CHANGER",
                action_params={"voltage": target_v},
                invariants=invariants,
                label="ALLOW",
                class_id=0
            )
        else:
            attack_type = self.rng.choice(["overvoltage", "overcurrent", "trip_hazard"])
            if attack_type == "overvoltage":
                bad_v = round(self.rng.choice([82.0, 142.5]), 1)
                return TraceSample(
                    domain="scada",
                    state=state,
                    action_name="INJECT_VOLTAGE_SPIKE",
                    action_params={"voltage": bad_v},
                    invariants=invariants,
                    label="BLOCK",
                    class_id=1,
                    breach_reason="GRID-VOLTAGE-STABILITY breached: voltage outside [90.0, 130.0] kV envelope"
                )
            else:
                surge_current = round(self.rng.uniform(850.0, 1400.0), 1)
                return TraceSample(
                    domain="scada",
                    state=state,
                    action_name="DISPATCH_OVERLOAD",
                    action_params={"current": surge_current},
                    invariants=invariants,
                    label="BLOCK",
                    class_id=1,
                    breach_reason="OVERCURRENT-PROTECTION breached: current surge exceeds 800.0 A ceiling"
                )

    def generate_dataset(self, total_samples: int = 50_000, adversarial_ratio: float = 0.35) -> List[TraceSample]:
        samples: List[TraceSample] = []
        for _ in range(total_samples):
            is_adv = (self.rng.random() < adversarial_ratio)
            domain = self.rng.choice(["treasury", "treasury", "scada"]) # 66% treasury, 33% scada
            if domain == "treasury":
                samples.append(self.generate_treasury_trace(is_adversarial=is_adv))
            else:
                samples.append(self.generate_scada_trace(is_adversarial=is_adv))
        return samples

    def export_jsonl(self, samples: List[TraceSample], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")
