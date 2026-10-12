# -*- coding: utf-8 -*-
"""
Single source of truth for the invariant rules.

Both the synthetic trace generator (to label samples) and the DecisionEngine's symbolic
oracle (to verify at runtime) call `oracle_decision`. Because the label is computed by the
same function on the same floats, generator and oracle are congruent by construction, and
the label can only depend on (state, params): never on the action name.
"""

from typing import Any, Dict, Optional, Tuple

MIN_OPERATING_LIQUIDITY_USD = 1_000_000.0
VOLTAGE_MIN_KV, VOLTAGE_MAX_KV = 90.0, 130.0
CURRENT_MAX_A = 800.0

LABEL_TO_CLASS = {"ALLOW": 0, "BLOCK": 1, "ROLLBACK": 2}


def oracle_decision(state: Dict[str, Any], action_name: str, params: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Returns (label, breach_reason). `action_name` is accepted for signature parity but never used."""

    # 0. Post-action state divergence (v0.2 only): the expected ledger checksum carried by the
    #    action no longer matches the ledger -> the action must be rolled back.
    if "expected_checksum" in params and "ledger_checksum" in state:
        if str(params["expected_checksum"]) != str(state["ledger_checksum"]):
            return "ROLLBACK", "STATE_DIVERGENCE: ledger checksum mismatch between action and committed state"

    # 1. Double-entry conservation on an explicit ledger entry.
    if "debit" in params and "credit" in params:
        if round(float(params["debit"]), 2) != round(float(params["credit"]), 2):
            return "BLOCK", "TREASURY-ZERO-SUM-CONSERVATION breached: debit and credit legs differ"
        return "ALLOW", None

    # 2. Transfers: sign constraint, then liquidity covenant on the source account.
    if "amount" in params:
        amount = float(params["amount"])
        if amount <= 0.0:
            return "BLOCK", "SIGN_CONSTRAINT violated: non-positive transaction amount"
        src = params.get("source_account")
        if src in state:
            min_liquidity = MIN_OPERATING_LIQUIDITY_USD if "operating_reserve" in src else 0.0
            if float(state[src]) - amount < min_liquidity:
                return "BLOCK", (f"MIN-OPERATING-LIQUIDITY-COVENANT violated: {src} would drop below "
                                 f"{min_liquidity:,.0f}")
        return "ALLOW", None

    # 3. SCADA envelopes.
    if "voltage" in params:
        v = float(params["voltage"])
        if v < VOLTAGE_MIN_KV or v > VOLTAGE_MAX_KV:
            return "BLOCK", f"GRID-VOLTAGE-STABILITY breached: voltage outside [{VOLTAGE_MIN_KV}, {VOLTAGE_MAX_KV}] kV"
        return "ALLOW", None

    if "current" in params:
        c = float(params["current"])
        if c > CURRENT_MAX_A or c < 0.0:
            return "BLOCK", f"OVERCURRENT-PROTECTION breached: current outside [0, {CURRENT_MAX_A}] A"
        return "ALLOW", None

    return "ALLOW", None
