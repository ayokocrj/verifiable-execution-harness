# -*- coding: utf-8 -*-
"""
Invariant Tokenizer: Domain-specific structured tokenizer for agent execution traces.
Encodes (State, Action, Criteria) triplets into integer token IDs.
"""

from typing import List, Dict, Any, Union

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]",
    "[STATE]", "[ACTION]", "[CRITERIA]",
    "[KEY]", "[VAL]", "[SEP]",
    "[ALLOW]", "[BLOCK]", "[ROLLBACK]"
]

class InvariantTokenizer:
    """
    Lightweight, deterministic tokenizer designed for structured systems invariants
    without requiring third-party tokenization dependencies (e.g. HuggingFace tokenizers).
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_base_vocabulary()

    def _build_base_vocabulary(self) -> None:
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # Pre-seed common financial, infrastructure and invariant terms
        common_terms = [
            "operating_reserve_usd", "payroll_clearing_usd", "vendor_escrow_usd",
            "treasury", "balance", "amount", "source_account", "dest_account",
            "SETTLE_PAYROLL", "EXCESSIVE_ESCROW_DRAIN", "FUND_VENDOR_ESCROW",
            "INJECT_UNBALANCED_CREDIT", "TRANSFER", "DRAIN", "CREDIT", "DEBIT",
            "voltage", "current", "grid_feeder", "transformer_1", "breaker_trip",
            "CONSERVATION", "RANGE_COVENANT", "MONOTONIC_AUDIT", "ZERO_SUM",
            "USD", "EUR", "GBP", "TRUE", "FALSE", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            ".", ":", ",", "-", "_", "{", "}", "[", "]", "=", ">", "<"
        ]

        idx = len(self.token_to_id)
        for term in common_terms:
            if term not in self.token_to_id:
                self.token_to_id[term] = idx
                self.id_to_token[idx] = term
                idx += 1

    def _tokenize_text(self, text: str) -> List[str]:
        # Normalize delimiters into discrete tokens
        normalized = text.replace("{", " { ").replace("}", " } ")
        normalized = normalized.replace("[", " [ ").replace("]", " ] ")
        normalized = normalized.replace(":", " : ").replace(",", " , ")
        normalized = normalized.replace("=", " = ").replace(">", " > ").replace("<", " < ")
        return [tok.strip() for tok in normalized.split() if tok.strip()]

    def encode(self, text: str, max_length: int = 128, add_special_tokens: bool = True) -> List[int]:
        raw_tokens = self._tokenize_text(text)
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.token_to_id["[BOS]"])

        for tok in raw_tokens:
            if tok in self.token_to_id:
                token_ids.append(self.token_to_id[tok])
            else:
                # Deterministic hash allocation for unseen terms within vocab_size
                hashed_id = len(SPECIAL_TOKENS) + (abs(hash(tok)) % (self.vocab_size - len(SPECIAL_TOKENS)))
                token_ids.append(hashed_id)

        if add_special_tokens:
            token_ids.append(self.token_to_id["[EOS]"])

        # Truncate if exceeds max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def pad(self, token_ids: List[int], max_length: int = 128) -> List[int]:
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        pad_id = self.token_to_id["[PAD]"]
        return token_ids + [pad_id] * (max_length - len(token_ids))

    def format_input(self, state: Dict[str, Any], action_name: str, params: Dict[str, Any], invariants: List[str]) -> str:
        """
        Formats structured inputs into a canonical representation:
        [STATE] state_dict [ACTION] action_name params [CRITERIA] invariant_list
        """
        state_str = " ".join([f"{k}:{v}" for k, v in sorted(state.items())])
        params_str = " ".join([f"{k}:{v}" for k, v in sorted(params.items())])
        criteria_str = " ".join(invariants)
        return f"[STATE] {state_str} [ACTION] {action_name} {params_str} [CRITERIA] {criteria_str}"
