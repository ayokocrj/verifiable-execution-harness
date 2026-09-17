# -*- coding: utf-8 -*-
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple

class VerifiableExecutionTrace:
    def __init__(
        self,
        step: int,
        previous_hash: str,
        action_type: str,
        parameters: Dict[str, Any],
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        rollback_executed: bool = False,
        violation_rule: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        self.step = step
        self.timestamp = timestamp or time.time()
        self.previous_hash = previous_hash
        self.action_type = action_type
        self.parameters = parameters
        self.state_before = state_before
        self.state_after = state_after
        self.rollback_executed = rollback_executed
        self.violation_rule = violation_rule
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = json.dumps({
            'step': self.step,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'state_before': self.state_before,
            'state_after': self.state_after,
            'rollback_executed': self.rollback_executed,
            'violation_rule': self.violation_rule
        }, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'hash': self.hash,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'state_before': self.state_before,
            'state_after': self.state_after,
            'rollback_executed': self.rollback_executed,
            'violation_rule': self.violation_rule
        }

class CryptographicLedger:
    GENESIS_HASH = '0000000000000000000000000000000000000000000000000000000000000000'

    def __init__(self):
        self.traces: List[VerifiableExecutionTrace] = []

    @property
    def latest_hash(self) -> str:
        return self.traces[-1].hash if self.traces else self.GENESIS_HASH

    def append_trace(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        rollback_executed: bool = False,
        violation_rule: Optional[str] = None
    ) -> VerifiableExecutionTrace:
        trace = VerifiableExecutionTrace(
            step=len(self.traces) + 1,
            previous_hash=self.latest_hash,
            action_type=action_type,
            parameters=parameters,
            state_before=state_before,
            state_after=state_after,
            rollback_executed=rollback_executed,
            violation_rule=violation_rule
        )
        self.traces.append(trace)
        return trace

    def audit(self) -> Tuple[bool, str]:
        if not self.traces:
            return True, 'Ledger is empty (genesis state intact).'
        for i, trace in enumerate(self.traces):
            if trace.hash != trace.compute_hash():
                return False, f'Integrity error at step {trace.step}: payload hash mismatch.'
            expected_prev = self.GENESIS_HASH if i == 0 else self.traces[i - 1].hash
            if trace.previous_hash != expected_prev:
                return False, f'Chain discontinuity at step {trace.step}.'
        return True, f'Audit PASSED: {len(self.traces)} blocks verified with 0 discrepancies.'

    def export_json(self) -> str:
        return json.dumps([t.to_dict() for t in self.traces], indent=2)
