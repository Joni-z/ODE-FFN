# Soft Lipschitz Small Experiment

## Goal

This experiment is meant to be a small, clean hypothesis check rather than a major new training recipe.

The project goal is still to modify JiT so it fits flow matching better from a dynamical-systems point of view.
This specific experiment asks a narrower question:

Does adding a very light smoothness / local Lipschitz constraint to the learned vector field help JiT behave more like a stable ODE-style flow model?

## Why Start With Loss Instead Of Architecture

Before changing JiT blocks directly, it is safer to test whether "smoother vector field" is even a useful signal.

If a light regularizer already shows no benefit, then a larger architecture change aimed at the same effect is less promising.
If it shows a useful trend, then we can later move toward more structural JiT changes that encode the same bias more naturally.

## Current Regularizer

We are not using the heavy full input-gradient penalty with `torch.autograd.grad(create_graph=True)`.

Instead, the current implementation uses a directional finite-difference approximation:

`||v_theta(z + eps * u, t) - v_theta(z, t)||^2 / eps^2`

where `u` is a random normalized direction.

This is closer to a local Lipschitz probe, but is much cheaper than building a full higher-order graph.

## Why The First Attempt Failed

The original run failed with CUDA OOM.

The main issue was not the formula itself, but the engineering cost:
the Lipschitz term required an extra forward pass through JiT.
With full batch size `256`, that extra pass was too expensive.

## Current "Small Experiment" Design

We want to keep the setup as close as possible to the `swiglu` baseline.

What should stay the same:

- JiT-B/16 architecture
- `swiglu` FFN
- batch size `256`
- standard training recipe as much as possible

What we allow to change:

- only the soft Lipschitz loss settings
- one practical memory-saving switch: `device_prefetch: false`

## Current YAML Intent

The current `configs/soft_lipschitz.yaml` is designed to be a light-touch variant of the baseline:

- `per_device_batch_size: 256`
- `device_prefetch: false`
- `soft_lipschitz.lambda: 1e-4`
- `soft_lipschitz.eps: 1e-2`
- `soft_lipschitz.num_samples: 8`

Interpretation:

- `lambda = 1e-4` keeps the regularizer weak so it does not dominate FM training.
- `num_samples = 8` means only a small subset of the batch is used for the extra perturbed forward pass.
- `device_prefetch = false` is only there to reduce avoidable GPU memory pressure.

## What This Experiment Is Trying To Measure

Main question:

- whether a light local smoothness bias helps training or sampling behavior

What to watch:

- `train/loss_lip` should be stable and nonzero
- training should run without immediate OOM
- FID should improve or at least not degrade badly versus `swiglu`
- low-NFE sampling may become a bit smoother or more stable

## Comparison Principle

When interpreting results, compare mainly against the plain `swiglu` baseline.

The logic should be:

baseline `swiglu`
vs.
`swiglu + very light soft Lipschitz`

This keeps the conclusion clean.

## If It Still OOMs

Do not change batch size first.

Use this fallback order:

1. reduce `soft_lipschitz.num_samples` from `8` to `4`
2. reduce `soft_lipschitz.lambda` from `1e-4` to `5e-5`
3. only then consider reducing batch size

Reason:

the goal is to preserve baseline comparability and only shrink the extra regularization cost.

## If The Experiment Shows Positive Signal

Then the next step should not be "make the penalty bigger" immediately.

A better next step is to move toward JiT-specific structural changes that reflect the same idea more naturally, for example:

- regularizing only the FFN residual branch
- adding a more explicit small-step / stable update bias inside JiT blocks
- constraining selected FFN projections instead of the whole model output

That would fit the long-term project goal better than relying on loss terms alone.

## Short Summary

This experiment is a probe, not the final method.

We are testing whether a slight local smoothness constraint helps JiT under flow matching.
If yes, that supports later architecture changes motivated by ODE-style stability.
If no, we avoid spending too much time on a large JiT redesign built on a weak premise.
