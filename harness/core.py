# -*- coding: utf-8 -*-
import copy
import time
from typing import Dict, Any, Callable
from .models import ActionRequest, ExecutionResult, InvariantViolationException
from .ledger import CryptographicLedger
from .invariants import InvariantEngine

class VerifiableAutonomyLayer:
    """
    Deterministic execution wrapper surrounding autonomous agent tool calls.
    Provides:
    1. Pre-execution spatial invariant checks.
    2. Isolated in-memory state transition sandbox.
    3. Post-execution boundary auditing.
    4. Deterministic state rollback on breach (revertible effects).
    5. Append-only cryptographic ledger (VETs).
    """

    def __init__(self, initial_state: Dict[str, Any]):
        self._state = copy.deepcopy(initial_state)
        self.ledger = CryptographicLedger()
        self.invariant_engine = InvariantEngine()

    @property
    def current_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._state)

    def add_invariant(self, invariant):
        self.invariant_engine.register(invariant)

    def execute(
        self,
        action: ActionRequest,
        mutator_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    ) -> ExecutionResult:
        t0 = time.perf_counter()
        state_before = copy.deepcopy(self._state)
        sandbox_state = copy.deepcopy(self._state)

        try:
            proposed_state = mutator_fn(sandbox_state, action.parameters)
            self.invariant_engine.evaluate(state_before, proposed_state)
            self._state = proposed_state
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            trace = self.ledger.append_trace(
                action_type=action.action_type,
                parameters=action.parameters,
                state_before=state_before,
                state_after=self._state,
                rollback_executed=False
            )

            return ExecutionResult(
                success=True,
                action_type=action.action_type,
                message=f"Action '{action.action_type}' committed. VET: {trace.hash[:16]}...",
                state_before=state_before,
                state_after=self._state,
                rollback_executed=False,
                execution_time_ms=t_elapsed_ms,
                vet_hash=trace.hash
            )

        except InvariantViolationException as e:
            self._state = state_before
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            trace = self.ledger.append_trace(
                action_type=action.action_type,
                parameters=action.parameters,
                state_before=state_before,
                state_after=state_before,
                rollback_executed=True,
                violation_rule=e.rule_name
            )

            return ExecutionResult(
                success=False,
                action_type=action.action_type,
                message=f"BREACH INTERCEPTED & ROLLED BACK: {e}",
                state_before=state_before,
                state_after=state_before,
                rollback_executed=True,
                execution_time_ms=t_elapsed_ms,
                vet_hash=trace.hash,
                violation_rule=e.rule_name
            )
