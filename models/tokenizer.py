# -*- coding: utf-8 -*-
"""
Invariant Tokenizer: Domain-specific structured tokenizer for agent execution traces.
Encodes (State, Action, Criteria) triplets into integer token IDs.

Design guarantees (all covered by tests/test_model.py):
  1. Deterministic across processes, machines and Python versions: unseen terms are
     mapped with CRC32, never with the salted built-in hash().
  2. Numeric scalars are split digit by digit ("598597.29" -> 5 9 8 5 9 7 . 2 9), so a
     model can learn magnitude and conservation invariants instead of seeing one opaque id.
  3. Special markers ([STATE], [ACTION], [CRITERIA], ...) are matched before any bracket
     normalisation and always receive their reserved id.
  4. Fallback ids never collide with the reserved base vocabulary.
"""

import re
import zlib
from typing import List, Dict, Any

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]",
    "[STATE]", "[ACTION]", "[CRITERIA]",
    "[KEY]", "[VAL]", "[SEP]",
    "[ALLOW]", "[BLOCK]", "[ROLLBACK]"
]

# Pre-seeded common financial, infrastructure and invariant terms.
COMMON_TERMS = [
    "operating_reserve_usd", "payroll_clearing_usd", "vendor_escrow_usd",
    "treasury", "balance", "amount", "source_account", "dest_account",
    "SETTLE_PAYROLL", "EXCESSIVE_ESCROW_DRAIN", "FUND_VENDOR_ESCROW",
    "INJECT_UNBALANCED_CREDIT", "TRANSFER", "DRAIN", "CREDIT", "DEBIT",
    "voltage", "current", "grid_feeder", "transformer_1", "breaker_trip",
    "CONSERVATION", "RANGE_COVENANT", "MONOTONIC_AUDIT", "ZERO_SUM",
    "USD", "EUR", "GBP", "TRUE", "FALSE", "True", "False", "None",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", ":", ",", "-", "_", "{", "}", "[", "]", "=", ">", "<", "(", ")", "'", '"'
]

# Longest special tokens first so "[STATE]" is never shadowed by "[".
_SPECIAL_PATTERN = "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True))

# Alternation order matters: special marker, signed number, identifier, any single non-space char.
# Identifiers may contain "-" so invariant names like TREASURY-ZERO-SUM-CONSERVATION stay whole.
_TOKEN_RE = re.compile(
    rf"(?P<special>{_SPECIAL_PATTERN})"
    r"|(?P<number>-?\d+(?:\.\d+)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_\-]*)"
    r"|(?P<punct>\S)"
)


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
        # Fallback ids live strictly above the base vocabulary.
        self.base_size = len(self.token_to_id)
        self.fallback_range = self.vocab_size - self.base_size
        if self.fallback_range <= 0:
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than the base vocabulary ({self.base_size})."
            )

    # ------------------------------------------------------------------ vocabulary
    def _build_base_vocabulary(self) -> None:
        for token in SPECIAL_TOKENS + COMMON_TERMS:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def _fallback_id(self, token: str) -> int:
        """Stable id for an out-of-vocabulary term. CRC32 is process- and platform-independent."""
        return self.base_size + (zlib.crc32(token.encode("utf-8")) % self.fallback_range)

    # ------------------------------------------------------------------ tokenisation
    @staticmethod
    def _split_number(number: str) -> List[str]:
        """'-598597.29' -> ['-', '5', '9', '8', '5', '9', '7', '.', '2', '9']"""
        return list(number)

    def _tokenize_text(self, text: str) -> List[str]:
        tokens: List[str] = []
        for m in _TOKEN_RE.finditer(text):
            kind = m.lastgroup
            tok = m.group()
            if kind == "number":
                tokens.extend(self._split_number(tok))
            else:
                tokens.append(tok)
        return tokens

    def convert_token_to_id(self, token: str) -> int:
        known = self.token_to_id.get(token)
        return known if known is not None else self._fallback_id(token)

    def encode(self, text: str, max_length: int = 128, add_special_tokens: bool = True) -> List[int]:
        token_ids: List[int] = [self.convert_token_to_id(t) for t in self._tokenize_text(text)]

        if add_special_tokens:
            bos, eos = self.token_to_id["[BOS]"], self.token_to_id["[EOS]"]
            # Truncate the body first so [EOS] always survives truncation.
            body_budget = max(0, max_length - 2)
            token_ids = [bos] + token_ids[:body_budget] + [eos]
        elif len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def pad(self, token_ids: List[int], max_length: int = 128) -> List[int]:
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        pad_id = self.token_to_id["[PAD]"]
        return token_ids + [pad_id] * (max_length - len(token_ids))

    def padding_mask(self, token_ids: List[int]) -> List[bool]:
        """True where the position is padding (matches torch key_padding_mask semantics)."""
        pad_id = self.token_to_id["[PAD]"]
        return [t == pad_id for t in token_ids]

    def format_input(self, state: Dict[str, Any], action_name: str, params: Dict[str, Any], invariants: List[str]) -> str:
        """
        Formats structured inputs into a canonical representation:
        [STATE] state_dict [ACTION] action_name params [CRITERIA] invariant_list
        """
        state_str = " ".join([f"{k}:{v}" for k, v in sorted(state.items())])
        params_str = " ".join([f"{k}:{v}" for k, v in sorted(params.items())])
        criteria_str = " ".join(invariants)
        return f"[STATE] {state_str} [ACTION] {action_name} {params_str} [CRITERIA] {criteria_str}"
