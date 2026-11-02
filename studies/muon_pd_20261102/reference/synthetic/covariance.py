"""Gradient fields with a signed isotropic part and a categorical covariance.

This is an internal-field hypothesis, not a constraint on the complete velocity
network. The untied and linearized variants are explicit mechanism controls.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class CovarianceField(nn.Module):
    def __init__(self, width: int, heads: int = 4, atoms: int = 8,
                 temperature: float = 1., mode: str = 'tied',
                 initial_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if width < 1 or heads < 1 or atoms < 2 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('invalid covariance field dimensions or temperature')
        if mode not in ('tied', 'untied', 'linearized'):
            raise ValueError('unsupported covariance field mode')
        self.width, self.heads, self.atoms = width, heads, atoms
        self.temperature, self.mode = temperature, mode
        if initial_weight is None:
            weight = torch.empty(heads * atoms, width)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            if initial_weight.shape != (heads * atoms, width):
                raise ValueError('initial encoder shape mismatch')
            weight = initial_weight.detach().clone()
        self.weight = nn.Parameter(weight)
        self.gamma = nn.Parameter(weight.new_zeros(()))
        if mode == 'untied':
            # No extra random draws: the initial hidden function matches tied.
            self.decoder = nn.Parameter(weight.t().clone())

    def decoder_weight(self) -> torch.Tensor:
        return self.decoder if self.mode == 'untied' else self.weight.t()

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.weight).reshape(*x.shape[:-1], self.heads, self.atoms)
        return torch.softmax(logits / self.temperature, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.weight).reshape(*x.shape[:-1], self.heads, self.atoms) / self.temperature
        if self.mode == 'linearized':
            centered = (logits - logits.mean(dim=-1, keepdim=True)) / self.atoms
        else:
            centered = torch.softmax(logits, dim=-1) - 1. / self.atoms
        return self.gamma * x + F.linear(centered.flatten(-2), self.decoder_weight())

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'untied':
            raise ValueError('the untied control has no guaranteed scalar potential')
        logits = F.linear(x, self.weight).reshape(*x.shape[:-1], self.heads, self.atoms)
        if self.mode == 'linearized':
            nonlinear = (logits - logits.mean(dim=-1, keepdim=True)).square().sum(dim=(-2, -1))
            nonlinear = nonlinear / (2 * self.atoms * self.temperature)
        else:
            nonlinear = (self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)
                         - logits.mean(dim=-1)).sum(dim=-1)
        return .5 * self.gamma * x.square().sum(dim=-1) + nonlinear

    def field_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'linearized':
            probabilities = x.new_full((*x.shape[:-1], self.heads, self.atoms), 1. / self.atoms)
        else:
            probabilities = self.probabilities(x)
        covariance = torch.diag_embed(probabilities) - probabilities.unsqueeze(-1) * probabilities.unsqueeze(-2)
        encoder = self.weight.reshape(self.heads, self.atoms, self.width)
        decoder = self.decoder_weight().reshape(self.width, self.heads, self.atoms)
        jacobian = torch.einsum('dhi,...hij,hje->...de', decoder, covariance, encoder) / self.temperature
        return jacobian + self.gamma * torch.eye(self.width, dtype=x.dtype, device=x.device)

    def linear_macs(self) -> int:
        return 2 * self.width * self.heads * self.atoms
