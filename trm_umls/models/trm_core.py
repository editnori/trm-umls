"""
TRM-UMLS core building blocks.

This module contains the shared TRM components used by the tiny text encoder:
- init helpers
- RMSNorm + SwiGLU
- TRMBlock (attention + FFN)
- config + carry state

The actual model lives in `trm_text_encoder.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


def trunc_normal_init_(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Truncated normal initialization."""
    with torch.no_grad():
        return tensor.normal_(0, std).clamp_(-2 * std, 2 * std)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMS normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit."""

    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inter = int(round(expansion * hidden_size * 2 / 3 / 256) * 256)
        if inter < 256:
            inter = 256

        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMBlock(nn.Module):
    """Single TRM block with attention (or optional sequence MLP) + FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float = 4.0,
        use_attention: bool = True,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.rms_norm_eps = rms_norm_eps

        if use_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True,
                bias=False,
            )
        else:
            self.seq_mlp = SwiGLU(hidden_size, expansion)

        self.ffn = SwiGLU(hidden_size, expansion)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            attn_out, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
            hidden_states = rms_norm(hidden_states + attn_out, self.rms_norm_eps)
        else:
            hidden_states = rms_norm(hidden_states + self.seq_mlp(hidden_states), self.rms_norm_eps)

        hidden_states = rms_norm(hidden_states + self.ffn(hidden_states), self.rms_norm_eps)
        return hidden_states


class TRMTextEncoderConfig(BaseModel):
    """Base configuration for TRM text encoding."""

    # Model architecture
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 2
    expansion: float = 4.0

    # Recursion parameters (key to TRM)
    h_cycles: int = 3  # High-level recursion cycles
    l_cycles: int = 4  # Low-level recursion cycles

    # Input/Output
    vocab_size: int = 30522  # BERT vocab size (will be updated based on tokenizer)
    max_seq_len: int = 128
    embedding_dim: int = 768  # Output dimension (matches current teacher space)

    # Training/runtime
    use_attention: bool = True
    rms_norm_eps: float = 1e-5

    class Config:
        extra = "allow"


@dataclass
class TRMCarry:
    """Carry state for recursive computation."""

    z_h: torch.Tensor  # High-level latent [batch, seq, hidden]
    z_l: torch.Tensor  # Low-level latent  [batch, seq, hidden]


class ContrastiveLoss(nn.Module):
    """InfoNCE loss for matching a teacher embedding space."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        pred_emb = F.normalize(pred_emb, p=2, dim=-1)
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        logits = torch.matmul(pred_emb, target_emb.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

