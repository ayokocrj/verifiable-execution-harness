# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .models import InvariantViolationException

class BaseInvariant(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def validate(self, state_before: Dict[str, Any], proposed_state: Dict[str, Any]) -> None:
        pass

class RangeInvariant(BaseInvariant):
    def __init__(self, name: str, field_name: str, min_val: float, max_val: float, unit: str = ""):
        super().__init__(name)
        self.field_name = field_name
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit

    def validate(self, state_before: Dict[str, Any], proposed_state: Dict[str, Any]) -> None:
        if self.field_name not in proposed_state:
            return
        val = proposed_state[self.field_name]
        if val < self.min_val or val > self.max_val:
            raise InvariantViolationException(
                self.name,
                f"Value {val}{self.unit} for '{self.field_name}' breached bounds [{self.min_val}, {self.max_val}{self.unit}]",
                proposed_state
            )

class MaxThresholdInvariant(BaseInvariant):
    def __init__(self, name: str, field_name: str, max_val: float, unit: str = ""):
        super().__init__(name)
        self.field_name = field_name
        self.max_val = max_val
        self.unit = unit

    def validate(self, state_before: Dict[str, Any], proposed_state: Dict[str, Any]) -> None:
        if self.field_name not in proposed_state:
            return
        val = proposed_state[self.field_name]
        if val > self.max_val:
            raise InvariantViolationException(
                self.name,
                f"Surge {val}{self.unit} for '{self.field_name}' exceeded maximum threshold {self.max_val}{self.unit}",
                proposed_state
            )

class ConservationInvariant(BaseInvariant):
    def __init__(self, name: str, balance_fields: List[str]):
        super().__init__(name)
        self.balance_fields = balance_fields

    def validate(self, state_before: Dict[str, Any], proposed_state: Dict[str, Any]) -> None:
        sum_before = sum(state_before.get(k, 0.0) for k in self.balance_fields)
        sum_after = sum(proposed_state.get(k, 0.0) for k in self.balance_fields)
        if abs(sum_before - sum_after) > 1e-6:
            raise InvariantViolationException(
                self.name,
                f"Conservation violated: total balance before ({sum_before}) != total balance after ({sum_after})",
                proposed_state
            )

class InvariantEngine:
    def __init__(self):
        self.invariants: List[BaseInvariant] = []

    def register(self, invariant: BaseInvariant):
        self.invariants.append(invariant)

    def evaluate(self, state_before: Dict[str, Any], proposed_state: Dict[str, Any]) -> None:
        for invariant in self.invariants:
            invariant.validate(state_before, proposed_state)
