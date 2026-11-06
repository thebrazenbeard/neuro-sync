"""CPU-only *toy* activation-transfer experiment, not an LLM runtime.

Two heterogeneous, frozen encoders produce representations of the same input.
A trainable projector maps source hidden states into target hidden space. A
target-local low-rank residual adapter and a bounded nonnegative gate allow
coupling using ordinary differentiable operations and autograd.

This intentionally does NOT splice gradients into unrelated parameters,
alias cross-architecture LoRA tensors, or claim emergent cognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARCHITECTURE = "neuro-sync-cpu-toy-v1"
INPUT_DIM = 4
SOURCE_DIM = 7
TARGET_DIM = 5
NUM_CLASSES = 2


class SourceEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 13), nn.GELU(),
            nn.Linear(13, SOURCE_DIM), nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TargetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 11), nn.Tanh(),
            nn.Linear(11, TARGET_DIM), nn.LayerNorm(TARGET_DIM),
        )
        self.head = nn.Linear(TARGET_DIM, NUM_CLASSES)

    def encode(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def classify(self, representation: Tensor) -> Tensor:
        return self.head(representation)


class TargetLocalAdapter(nn.Module):
    """Only target-width weights are adapted; no cross-model weight sharing."""

    def __init__(self, width: int = TARGET_DIM, rank: int = 3) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=0.05)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


@dataclass(frozen=True)
class ForwardState:
    logits: Tensor
    target_hidden: Tensor
    projected_source: Tensor
    effective_gate: Tensor


class NeuroSyncToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source = SourceEncoder()
        self.target = TargetBackbone()
        for backbone in (self.source, self.target):
            backbone.requires_grad_(False)

        # P : R^7 -> R^5; no false claim of P^T P = I.
        self.projector = nn.Linear(SOURCE_DIM, TARGET_DIM, bias=False)
        self.adapter = TargetLocalAdapter()
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _validate_input(x: Tensor, alpha: float) -> None:
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise ValueError(f"expected input [batch, {INPUT_DIM}], got {tuple(x.shape)}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1 inclusive")

    def baseline_logits(self, x: Tensor) -> Tensor:
        """Same frozen target and local adapter, with no source coupling."""
        self._validate_input(x, 0.0)
        with torch.no_grad():
            target_hidden = self.target.encode(x)
        return self.target.classify(target_hidden + self.adapter(target_hidden))

    def forward(
        self, x: Tensor, *, alpha: float = 1.0, shuffle_source: bool = False
    ) -> ForwardState:
        self._validate_input(x, alpha)
        if shuffle_source and x.shape[0] < 2:
            raise ValueError("shuffled transfer requires batch size >= 2")

        with torch.no_grad():
            source_hidden = self.source(x)
            target_hidden = self.target.encode(x)
        if shuffle_source:
            # Explicit negative control: break correspondence without changing
            # target inputs or labels. No stochastic pairing ambiguity.
            source_hidden = source_hidden.roll(shifts=1, dims=0)

        projected = self.projector(source_hidden)
        effective_gate = alpha * torch.sigmoid(self.gate_logit)  # [0, alpha]
        fused = target_hidden + self.adapter(target_hidden) + effective_gate * projected
        return ForwardState(
            logits=self.target.classify(fused),
            target_hidden=target_hidden,
            projected_source=projected,
            effective_gate=effective_gate,
        )

    def loss(
        self,
        x: Tensor,
        labels: Tensor,
        *,
        alpha: float = 1.0,
        shuffle_source: bool = False,
        alignment_weight: float = 0.1,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if labels.ndim != 1 or labels.shape[0] != x.shape[0]:
            raise ValueError("labels must be [batch] and match input batch size")
        if labels.dtype != torch.long:
            raise ValueError("labels must use torch.long")
        if alignment_weight < 0:
            raise ValueError("alignment_weight must be nonnegative")

        state = self.forward(x, alpha=alpha, shuffle_source=shuffle_source)
        task = F.cross_entropy(state.logits, labels)
        # alpha=0 exactly removes all transfer objectives and their gradients.
        alignment = alpha * alignment_weight * F.mse_loss(
            state.projected_source, state.target_hidden.detach()
        )
        return task + alignment, {"task": task, "alignment": alignment}


def make_synthetic_batch(batch_size: int = 32, seed: int = 11) -> tuple[Tensor, Tensor]:
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, INPUT_DIM, generator=generator)
    labels = (1.7 * x[:, 0] - x[:, 1] + 0.4 * x[:, 2].square() > 0.3).long()
    return x, labels


def train_steps(
    model: NeuroSyncToy,
    x: Tensor,
    labels: Tensor,
    *,
    steps: int = 10,
    alpha: float = 1.0,
    shuffle_source: bool = False,
) -> list[float]:
    """Return observed training losses; no assertion that transfer helps."""
    if steps < 1:
        raise ValueError("steps must be positive")
    model.train()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=0.01)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(x, labels, alpha=alpha, shuffle_source=shuffle_source)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        for parameter in model.trainable_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("missing or non-finite trainable gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def save_checkpoint(model: NeuroSyncToy, path: str | Path) -> None:
    torch.save({"architecture": ARCHITECTURE, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path) -> NeuroSyncToy:
    # Only load checkpoints from a trusted origin. weights_only blocks ordinary
    # pickle object reconstruction on supported PyTorch versions.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("incompatible checkpoint architecture")
    model = NeuroSyncToy()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


if __name__ == "__main__":
    for name, alpha, shuffled in (
        ("local-only", 0.0, False),
        ("aligned-transfer", 1.0, False),
        ("shuffled-control", 1.0, True),
    ):
        torch.manual_seed(17)  # matched initialization across conditions
        candidate = NeuroSyncToy()
        features, truth = make_synthetic_batch()
        losses = train_steps(candidate, features, truth, alpha=alpha, shuffle_source=shuffled)
        print(f"{name}: initial={losses[0]:.6f} final={losses[-1]:.6f}")
