"""RWKV7 (G1j) architecture config.

Values match the official RWKV7-G1j checkpoints (e.g. G1j-2.9B):
see ``config.json`` of the HF-format export and the FLA reference
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RWKV7Config:
    vocab_size: int = 65536
    hidden_size: int = 2560
    num_heads: int = 40
    head_dim: int = 64
    num_layers: int = 32
    intermediate_size: int = 10240
    # low-rank dims
    decay_low_rank_dim: int = 96   # w (time decay)
    a_low_rank_dim: int = 96       # a (iclr)
    v_low_rank_dim: int = 64       # value residual mix
    gate_low_rank_dim: int = 320   # output gate g
    # norms
    norm_eps: float = 1e-5
    norm_bias: bool = True
    norm_first: bool = True
    hidden_act: str = "sqrelu"
    # Wkv constants
    w_scale: float = -0.6065306597126334  # w = exp(w_scale * sigmoid(w_lora(xw)))

    @property
    def key_dim(self) -> int:
        return self.hidden_size

    @property
    def value_dim(self) -> int:
        return self.hidden_size

    @property
    def group_norm_eps(self) -> float:
        return self.head_dim * self.norm_eps

    @property
    def state_numel(self) -> int:
        """Wkv state elements per layer: num_heads * head_dim * head_dim.

        The RWKV7 Wkv state is a full head_dim x head_dim matrix per head.
        """
        return self.num_heads * self.head_dim * self.head_dim

    @classmethod
    def from_dict(cls, d: dict):
        """Build from an HF config.json (tolerates missing optional keys)."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        mapped = {k: v for k, v in d.items() if k in known}
        # HF configs name the depth `num_hidden_layers`; normalize.
        if "num_hidden_layers" in d and "num_layers" not in mapped:
            mapped["num_layers"] = d["num_hidden_layers"]
        return cls(**mapped)
