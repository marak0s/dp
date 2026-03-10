from __future__ import annotations

import torch
from torch import nn


def bce_logits_loss(logits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(logits, target_bits)
