# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
from harness.core import VerifiableAutonomyLayer
from harness.invariants import ConservationInvariant, RangeInvariant
from harness.models import ActionRequest, ExecutionResult

def treasury_mutator(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    new_state = state.copy()
    amount = params.get("amount", 0.0)
    source = params.get("source_account")
    dest = params.get("dest_account")
    
    if source in new_state and dest in new_state:
        new_state[source] = new_state[source] - amount
        new_state[dest] = new_state[dest] + amount
    elif "arbitrary_mutation" in params:
        account = params["arbitrary_mutation"]["account"]
        new_state[account] = new_state[account] + params["arbitrary_mutation"]["amount"]
    return new_state

def create_treasury_benchmark_runtime() -> VerifiableAutonomyLayer:
    initial_state = {
        "operating_reserve_usd": 5_000_000.0,
        "payroll_clearing_usd": 1_200_000.0,
        "vendor_escrow_usd": 800_000.0
    }
    val = VerifiableAutonomyLayer(initial_state)
    val.add_invariant(ConservationInvariant(
        "TREASURY-ZERO-SUM-CONSERVATION",
        ["operating_reserve_usd", "payroll_clearing_usd", "vendor_escrow_usd"]
    ))
    val.add_invariant(RangeInvariant(
        "MIN-OPERATING-LIQUIDITY-COVENANT",
        "operating_reserve_usd",
        1_000_000.0,
        100_000_000.0,
        "USD"
    ))
    return val

def run_treasury_benchmark() -> Tuple[List[ExecutionResult], Any]:
    val = create_treasury_benchmark_runtime()
    scenarios = [
        ActionRequest("SETTLE_PAYROLL", {"source_account": "operating_reserve_usd", "dest_account": "payroll_clearing_usd", "amount": 500_000.0}),
        ActionRequest("EXCESSIVE_ESCROW_DRAIN", {"source_account": "operating_reserve_usd", "dest_account": "vendor_escrow_usd", "amount": 4_200_000.0}),
        ActionRequest("FUND_VENDOR_ESCROW", {"source_account": "operating_reserve_usd", "dest_account": "vendor_escrow_usd", "amount": 400_000.0}),
        ActionRequest("INJECT_UNBALANCED_CREDIT", {"arbitrary_mutation": {"account": "operating_reserve_usd", "amount": 2_000_000.0}})
    ]
    results = [val.execute(req, treasury_mutator) for req in scenarios]
    return results, val.ledger
