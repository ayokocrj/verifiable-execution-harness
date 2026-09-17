# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

@dataclass
class ActionRequest:
    action_type: str
    parameters: Dict[str, Any]
    caller_id: str = 'agent-executor'
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExecutionResult:
    success: bool
    action_type: str
    message: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    rollback_executed: bool = False
    execution_time_ms: float = 0.0
    vet_hash: Optional[str] = None
    violation_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class InvariantViolationException(Exception):
    def __init__(self, rule_name: str, message: str, proposed_state: Dict[str, Any]):
        super().__init__(f'[{rule_name}] {message}')
        self.rule_name = rule_name
        self.message = message
        self.proposed_state = proposed_state
