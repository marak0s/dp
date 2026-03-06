from __future__ import annotations

from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from comm_ai.decoders.neural_bcjr import NeuralBCJRDecoder


def train_neural_bcjr(model: NeuralBCJRDecoder, llr: torch.Tensor, target: torch.Tensor, epochs: int = 3, lr: float = 1e-3, save_path: str | Path | None = None) -> NeuralBCJRDecoder:
    opt = optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    for _ in tqdm(range(epochs), desc="train_neural_bcjr"):
        opt.zero_grad()
        out = model.calibrate(llr)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        if loss.item() < best and save_path is not None:
            best = loss.item()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
    return model
