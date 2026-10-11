# -*- coding: utf-8 -*-
"""
DecisionEngine: Runtime inference orchestrator for Ledger-Invariant verification.
Connects the Verifiable Autonomy Layer (VAL) with compact neural decision verifiers.
"""

import time
from typing import Dict, Any, List, Tuple
from models.tokenizer import InvariantTokenizer
from models.architecture import LedgerInvariantConfig, LedgerInvariantDecisionTransformer, TORCH_AVAILABLE

DECISION_CLASSES = {0: "ALLOW", 1: "BLOCK", 2: "ROLLBACK"}

class DecisionResult:
    def __init__(self, decision: str, class_id: int, confidence: float, latency_ms: float, details: Dict[str, Any]):
        self.decision = decision          # "ALLOW", "BLOCK", "ROLLBACK"
        self.class_id = class_id          # 0, 1, 2
        self.confidence = confidence      # 0.0 - 1.0
        self.latency_ms = latency_ms      # runtime latency in milliseconds
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 3),
            "details": self.details
        }

class DecisionEngine:
    def __init__(self, config: LedgerInvariantConfig = None, model_path: str = None):
        self.config = config or LedgerInvariantConfig.get_35m_config()
        self.tokenizer = InvariantTokenizer(vocab_size=self.config.vocab_size)
        self.model = None
        self.has_weights = False

        if TORCH_AVAILABLE and model_path:
            import torch
            self.model = LedgerInvariantDecisionTransformer(self.config)
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.has_weights = True
            except Exception:
                self.has_weights = False

    def predict(
        self,
        current_state: Dict[str, Any],
        action_name: str,
        action_params: Dict[str, Any],
        invariant_names: List[str]
    ) -> DecisionResult:
        """
        Evaluates a proposed state mutation and returns a deterministic decision.
        """
        t0 = time.perf_counter()

        formatted_input = self.tokenizer.format_input(
            current_state, action_name, action_params, invariant_names
        )
        token_ids = self.tokenizer.encode(formatted_input, max_length=self.config.max_seq_len)

        # If neural model weights are loaded and torch is active, run model forward
        if TORCH_AVAILABLE and self.has_weights and self.model is not None:
            import torch
            with torch.no_grad():
                input_tensor = torch.tensor([token_ids], dtype=torch.long)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                class_id = torch.argmax(probs).item()
                confidence = probs[class_id].item()
                decision = DECISION_CLASSES.get(class_id, "BLOCK")
        else:
            # Deterministic symbolic oracle baseline evaluation
            decision, class_id, confidence = self._symbolic_evaluator(
                current_state, action_name, action_params, invariant_names
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            decision=decision,
            class_id=class_id,
            confidence=confidence,
            latency_ms=latency_ms,
            details={"tokens_count": len(token_ids), "mode": "neural" if self.has_weights else "symbolic_oracle"}
        )

    def _symbolic_evaluator(
        self,
        state: Dict[str, Any],
        action: str,
        params: Dict[str, Any],
        invariants: List[str]
    ) -> Tuple[str, int, float]:
        """
        Deterministic mathematical invariant evaluator for zero-shot testing and ground-truth validation.
        """
        amount = params.get("amount", 0.0)

        # 1. Negative amounts or arbitrary mutations outside double-entry balance -> BLOCK
        if amount < 0 or "arbitrary_mutation" in params:
            return "BLOCK", 1, 0.999

        # 2. Financial treasury domain checks
        if "source_account" in params and "dest_account" in params:
            src = params["source_account"]
            dst = params["dest_account"]

            if src in state:
                # Overdraft or minimum covenant breach
                min_liquidity = 1_000_000.0 if "operating_reserve" in src else 0.0
                if state[src] - amount < min_liquidity:
                    return "BLOCK", 1, 0.985
                return "ALLOW", 0, 0.992

        # 3. SCADA substation domain checks
        if "voltage" in params:
            v = params["voltage"]
            if v < 90.0 or v > 130.0:
                return "BLOCK", 1, 0.998
            return "ALLOW", 0, 0.995

        if "current" in params:
            c = params["current"]
            if c > 800.0 or c < 0.0:
                return "BLOCK", 1, 0.998
            return "ALLOW", 0, 0.995

        return "ALLOW", 0, 0.900
