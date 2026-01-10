"""
TRM Text Encoder: multi-task tiny recursive text encoder.

Outputs:
- `embedding`: vector used for UMLS concept matching (FAISS lookup)
- `assertion_logits`: PRESENT/ABSENT/POSSIBLE (optional)
- `subject_logits`: PATIENT/FAMILY/OTHER (optional)

Note: TUI / semantic type comes from the CUI lookup after retrieval.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trm_core import (
    TRMBlock,
    TRMCarry,
    TRMTextEncoderConfig as _BaseConfig,
    trunc_normal_init_,
)


class TRMTextEncoderConfig(_BaseConfig):
    """Configuration for the multi-task TRM Text Encoder."""

    # Classification heads
    num_assertion_labels: int = 3  # PRESENT, ABSENT, POSSIBLE
    num_subject_labels: int = 3  # PATIENT, FAMILY, OTHER

    # Head dimensions
    classifier_hidden_size: int = 128
    classifier_dropout: float = 0.1


class ClassificationHead(nn.Module):
    """Classification head with dropout and hidden layer."""

    def __init__(self, input_size: int, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.classifier(x)


class TRMTextEncoder(nn.Module):
    """
    Multi-task TRM Text Encoder.

    Produces:
    - Embeddings for UMLS concept matching
    - Assertion classification (PRESENT/ABSENT/POSSIBLE)
    - Subject classification (PATIENT/FAMILY/OTHER)
    """

    def __init__(self, config: TRMTextEncoderConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size)

        # Position embedding (learned)
        self.embed_pos = nn.Embedding(config.max_seq_len, config.hidden_size)

        # Reasoning layers (shared encoder)
        self.layers = nn.ModuleList(
            [
                TRMBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    expansion=config.expansion,
                    use_attention=config.use_attention,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Initial states for recursion
        self.h_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size), std=1.0))
        self.l_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size), std=1.0))

        # Embedding projection head (for UMLS matching)
        self.embedding_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.embedding_dim),
        )

        # Assertion classification head (PRESENT/ABSENT/POSSIBLE)
        self.assertion_head = ClassificationHead(
            input_size=config.hidden_size,
            hidden_size=config.classifier_hidden_size,
            num_labels=config.num_assertion_labels,
            dropout=config.classifier_dropout,
        )

        # Subject classification head (PATIENT/FAMILY/OTHER)
        self.subject_head = ClassificationHead(
            input_size=config.hidden_size,
            hidden_size=config.classifier_hidden_size,
            num_labels=config.num_subject_labels,
            dropout=config.classifier_dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_init_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                trunc_normal_init_(module.weight, std=0.02)

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        token_emb = self.embed_tokens(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.embed_pos(positions)
        return self.embed_scale * token_emb + pos_emb

    def _init_carry(self, batch_size: int, seq_len: int, device: torch.device) -> TRMCarry:
        return TRMCarry(
            z_h=self.h_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1),
            z_l=self.l_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1),
        )

    def _reasoning_step(self, carry: TRMCarry, input_emb: torch.Tensor) -> TRMCarry:
        z_h, z_l = carry.z_h, carry.z_l

        for _h in range(self.config.h_cycles):
            for _l in range(self.config.l_cycles):
                z_l_input = z_l + z_h + input_emb
                for layer in self.layers:
                    z_l_input = layer(z_l_input)
                z_l = z_l_input

            z_h_input = z_h + z_l
            for layer in self.layers:
                z_h_input = layer(z_h_input)
            z_h = z_h_input

        return TRMCarry(z_h=z_h, z_l=z_l)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to pooled hidden representation [batch, hidden_size]."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        input_emb = self._input_embeddings(input_ids)
        carry = self._init_carry(batch_size, seq_len, device)
        carry = self._reasoning_step(carry, input_emb)

        hidden = carry.z_h
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            hidden = hidden.mean(dim=1)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.

        task:
          - "all": embedding + logits
          - "embedding": only embedding
          - "assertion": only assertion logits
          - "subject": only subject logits
        """
        hidden = self.encode(input_ids, attention_mask)
        outputs: Dict[str, torch.Tensor] = {}

        if task in ("all", "embedding"):
            emb = self.embedding_head(hidden)
            outputs["embedding"] = F.normalize(emb, p=2, dim=-1)

        if task in ("all", "assertion"):
            outputs["assertion_logits"] = self.assertion_head(hidden)

        if task in ("all", "subject"):
            outputs["subject_logits"] = self.subject_head(hidden)

        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained_v1(cls, checkpoint_path: Path, config: TRMTextEncoderConfig) -> "TRMTextEncoder":
        """
        Load from a legacy embedding-only checkpoint and add new heads.

        This allows continuing training from an older embedding model checkpoint while
        adding classification heads.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = cls(config)

        v1_state = checkpoint["model_state_dict"]
        model_state = model.state_dict()

        for name, param in v1_state.items():
            new_name = name.replace("projection.", "embedding_head.")
            if new_name in model_state and model_state[new_name].shape == param.shape:
                model_state[new_name] = param

        model.load_state_dict(model_state)
        return model


ASSERTION_LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]
SUBJECT_LABELS = ["PATIENT", "FAMILY", "OTHER"]


if __name__ == "__main__":
    config = TRMTextEncoderConfig(
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        h_cycles=3,
        l_cycles=4,
        embedding_dim=768,
        num_assertion_labels=3,
        num_subject_labels=3,
    )
    model = TRMTextEncoder(config)
    print(f"TRM Text Encoder")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Output dim: {config.embedding_dim}")
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, task="all")
    for k, v in outputs.items():
        print(f"  {k}: {tuple(v.shape)}")

