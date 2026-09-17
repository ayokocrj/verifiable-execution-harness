# -*- coding: utf-8 -*-
"""
Simplified Power Substation Model (Illustrative Physical Bounds).
Conceptually inspired by electrical grid transmission safety envelopes.
"""
from typing import Dict, Any, List, Tuple
from harness.core import VerifiableAutonomyLayer
from harness.invariants import RangeInvariant, MaxThresholdInvariant
from harness.models import ActionRequest, ExecutionResult

def scada_state_mutator(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    new_state = state.copy()
    for k in ["voltage_kv", "current_amps", "breaker_closed", "tap_position"]:
        if k in params:
            new_state[k] = params[k]
    return new_state

def create_scada_benchmark_runtime() -> VerifiableAutonomyLayer:
    initial_state = {
        "substation_id": "SUB-NORTH-400",
        "voltage_kv": 110.0,
        "current_amps": 450.0,
        "breaker_closed": True,
        "tap_position": 5
    }
    val = VerifiableAutonomyLayer(initial_state)
    val.add_invariant(RangeInvariant("SUBSTATION-VOLT-ENVELOPE", "voltage_kv", 90.0, 130.0, "kV"))
    val.add_invariant(MaxThresholdInvariant("SUBSTATION-MAX-CURRENT", "current_amps", 800.0, "A"))
    return val

def run_scada_benchmark() -> Tuple[List[ExecutionResult], Any]:
    val = create_scada_benchmark_runtime()
    scenarios = [
        ActionRequest("ADJUST_TAP", {"voltage_kv": 114.5, "tap_position": 6}),
        ActionRequest("LOAD_SURGE_INJECTION", {"current_amps": 920.0}),
        ActionRequest("REGULATE_LOAD", {"current_amps": 520.0}),
        ActionRequest("VOLTAGE_SPIKE", {"voltage_kv": 142.0}),
        ActionRequest("ISOLATE_FEEDER", {"breaker_closed": False, "current_amps": 0.0})
    ]
    results = [val.execute(req, scada_state_mutator) for req in scenarios]
    return results, val.ledger
