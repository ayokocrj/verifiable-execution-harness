# -*- coding: utf-8 -*-
"""
LedgerInvariantDecisionTransformer (10M to 35M parameters)
Custom compact transformer architecture for deterministic decision scoring.
Evaluates (State, Action, Criteria) and outputs logits across [ALLOW, BLOCK, ROLLBACK].
"""

import math
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Graceful fallback shim for environments without torch installed locally
    class nn:
        class Module:
            def __init__(self): pass
            def parameters(self): return []
            def forward(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)

class LedgerInvariantConfig:
    """Configuration hyperparameters for the Ledger-Invariant Decision Model family."""
    def __init__(
        self,
        vocab_size: int = 4096,
        max_seq_len: int = 256,
        d_model: int = 384,          # 384 -> ~10M params (4 layers) to ~35M params (8-12 layers)
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        num_classes: int = 3         # 0: ALLOW, 1: BLOCK, 2: ROLLBACK
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_classes = num_classes

    FIELDS = ("vocab_size", "max_seq_len", "d_model", "n_heads", "n_layers", "d_ff", "dropout", "num_classes")

    @classmethod
    def get_10m_config(cls, num_classes: int = 2) -> "LedgerInvariantConfig":
        """v0.1: binary pre-execution gatekeeper (ALLOW / BLOCK) by default.
        Generated traces are at most ~70 tokens, so a 128-token window keeps padding under 50%."""
        return cls(d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_seq_len=128, num_classes=num_classes)

    @classmethod
    def get_35m_config(cls, num_classes: int = 3) -> "LedgerInvariantConfig":
        """v0.2: adds the post-action ROLLBACK class."""
        return cls(d_model=512, n_heads=8, n_layers=10, d_ff=2048, num_classes=num_classes)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LedgerInvariantConfig":
        return cls(**{k: d[k] for k in cls.FIELDS if k in d})

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LedgerInvariantConfig":
        import json
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def count_parameters(self) -> int:
        """Mathematical estimation of parameter count."""
        # Token embeddings + Positional embeddings
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        # Per layer: Self-attention (4 * d_model^2) + FFN (2 * d_model * d_ff) + LayerNorms
        layer_params = (4 * (self.d_model ** 2)) + (2 * self.d_model * self.d_ff) + (4 * self.d_model)
        # Decision Head: LayerNorm + Projection (d_model * num_classes)
        head_params = (2 * self.d_model) + (self.d_model * self.num_classes)
        return emb + (self.n_layers * layer_params) + head_params


if TORCH_AVAILABLE:
    class TransformerBlock(nn.Module):
        def __init__(self, config: LedgerInvariantConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(config.d_model)
            self.attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                dropout=config.dropout,
                batch_first=True
            )
            self.ln2 = nn.LayerNorm(config.d_model)
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout)
            )

        def forward(self, x, mask=None):
            norm_x = self.ln1(x)
            attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=mask)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x

    class DecisionClassificationHead(nn.Module):
        """
        Structured Decision Head:
        Projects the contextual representation of the [BOS] token into 3 calibrated decision logits:
        [ALLOW, BLOCK, ROLLBACK]
        """
        def __init__(self, d_model: int, num_classes: int = 3, dropout: float = 0.1):
            super().__init__()
            self.dense = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.out_proj = nn.Linear(d_model, num_classes)

        def forward(self, features):
            # Take the first token representation [BOS]
            x = features[:, 0, :]
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            logits = self.out_proj(x)
            return logits

    class LedgerInvariantDecisionTransformer(nn.Module):
        def __init__(self, config: LedgerInvariantConfig):
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
            self.drop = nn.Dropout(config.dropout)
            
            self.blocks = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_layers)
            ])
            self.ln_f = nn.LayerNorm(config.d_model)
            self.decision_head = DecisionClassificationHead(
                config.d_model, config.num_classes, config.dropout
            )

        def forward(self, input_ids, padding_mask=None):
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            
            x = self.token_embedding(input_ids) + self.pos_embedding(positions)
            x = self.drop(x)

            for block in self.blocks:
                x = block(x, mask=padding_mask)

            x = self.ln_f(x)
            logits = self.decision_head(x)
            return logits

else:
    # Emulated interface when running outside a PyTorch installation
    class LedgerInvariantDecisionTransformer(nn.Module):
        def __init__(self, config: LedgerInvariantConfig):
            self.config = config

        def forward(self, input_ids, padding_mask=None):
            raise NotImplementedError(
                "PyTorch is not installed in the current environment. "
                "Install PyTorch (pip install torch) to run neural forward execution."
            )
