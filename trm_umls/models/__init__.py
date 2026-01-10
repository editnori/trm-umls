"""TRM-UMLS models (default encoder + shared core blocks)."""

from .trm_text_encoder import TRMTextEncoder, TRMTextEncoderConfig
from .trm_core import ContrastiveLoss

__all__ = ["TRMTextEncoder", "TRMTextEncoderConfig", "ContrastiveLoss"]
