"""Fresh analytic-population inputs and optional conditional-mean labels.

These are diagnostic training controls. Both supervision choices use exactly
the same population/input stream; an analytic teacher is not a new architecture.
"""
from __future__ import annotations

import hashlib
import torch

from synthetic.data import GaussianMixture, RotatedGaussianMixture, RotatedProductMixture


class PopulationStream:
    def __init__(self, distribution, device: torch.device | str = 'cpu') -> None:
        if not isinstance(distribution,(GaussianMixture,RotatedGaussianMixture,RotatedProductMixture)):
            raise ValueError('fresh population control requires an analytic mixture')
        self.device = torch.device(device)
        self.dim = distribution.dim
        self.product = isinstance(distribution, RotatedProductMixture)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=self.device)
        self.rotation = None
        if self.product:
            self.locations = tensor(distribution.locations)
            self.variances = tensor(distribution.variances)
            self.rotation = tensor(distribution.rotation)
        else:
            if isinstance(distribution, RotatedGaussianMixture):
                self.rotation = tensor(distribution.rotation)
                distribution = distribution.base
            if not isinstance(distribution, GaussianMixture):
                raise ValueError('fresh population control requires an analytic mixture')
            self.means = tensor(distribution.means)
            self.variances = tensor(distribution.variances)
            self.weights = tensor(distribution.weights)

    @torch.no_grad()
    def sample(self, count: int, generator: torch.Generator) -> torch.Tensor:
        if self.product:
            signs = 2 * torch.randint(0, 2, (count, self.dim), generator=generator, device=self.device) - 1
            noise = torch.randn((count, self.dim), dtype=torch.float64, generator=generator, device=self.device)
            target = signs * self.locations + self.variances.sqrt() * noise
        else:
            component = torch.multinomial(self.weights, count, replacement=True, generator=generator)
            noise = torch.randn((count, self.dim), dtype=torch.float64, generator=generator, device=self.device)
            target = self.means[component] + self.variances[component].sqrt() * noise
        if self.rotation is not None:
            target = target @ self.rotation.T
        return target.float()

    @torch.no_grad()
    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim or t.shape != (len(x), 1):
            raise ValueError('expected x=[batch,dim] and t=[batch,1]')
        original_dtype = x.dtype
        latent, time = x.double(), t.double()
        if self.rotation is not None:
            latent = latent @ self.rotation
        if self.product:
            variance = (1-time).square() + time.square() * self.variances
            coefficient = (time * self.variances - (1-time)) / variance
            argument = time * self.locations * latent / variance
            value = coefficient * latent + (1-time) / variance * self.locations * argument.tanh()
        else:
            time = time.unsqueeze(-1)
            variance = (1-time).square() + time.square() * self.variances
            residual = latent[:, None] - time * self.means
            logits = self.weights.log() - .5 * (variance.log() + residual.square() / variance).sum(dim=-1)
            responsibility = torch.softmax(logits, dim=-1)
            component_velocity = self.means + (time * self.variances - (1-time)) / variance * residual
            value = (responsibility[:, :, None] * component_velocity).sum(dim=1)
        if self.rotation is not None:
            value = value @ self.rotation.T
        return value.to(original_dtype)


def fingerprint_batch(previous: str, target: torch.Tensor, source: torch.Tensor, time: torch.Tensor) -> str:
    """Hash-chain the actual float32 sampled inputs, excluding supervision choice."""
    digest = hashlib.sha256(bytes.fromhex(previous))
    for tag, value in [(b'target', target), (b'source', source), (b'time', time)]:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(tag)
        digest.update(str((array.shape, str(array.dtype))).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()
