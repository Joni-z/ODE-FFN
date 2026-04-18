# Frequency Split Branch

## Motivation

This branch explores a frequency-aware FFN for JiT.

The idea is not to claim that a full FFT-based architecture is required.
Instead, the goal is to test a smaller structural hypothesis:

- low-frequency content and high-frequency detail may benefit from different FFN sub-functions
- JiT currently uses a single FFN per block, which may be too uniform for both coarse structure and fine detail
- a lightweight dual-path FFN can test this without rewriting the whole model

This direction is motivated by several references:

1. Spectral bias:
   Rahaman et al., "On the Spectral Bias of Neural Networks"
   https://arxiv.org/abs/1806.08734

2. ViT low-pass tendency:
   Park and Kim, "Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis"
   https://arxiv.org/abs/2203.05962

3. High/low frequency branch design:
   Chen et al., "Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution"
   https://arxiv.org/abs/1904.05049

4. Lightweight frequency statistics inside standard modules:
   Qin et al., "FcaNet: Frequency Channel Attention Networks"
   https://arxiv.org/abs/2012.11879

These papers do not directly give a JiT flow-matching architecture, but together they support a reasonable branch hypothesis:

- neural nets often fit low frequencies more easily than high frequencies
- deep vision backbones can behave like low-pass operators
- splitting or routing by frequency is a legitimate structural prior

## First Implementation

The first implementation is intentionally conservative and stays inside the existing FFN framework.

Name:

- `freq_split`

Placement:

- only the FFN is changed
- the JiT backbone, attention stack, and training recipe remain unchanged

Structure:

1. Build a local low-pass token stream with a small spatial average pool on the token grid
2. Define the high-frequency token stream as `x - lowpass(x)`
3. Send low-pass tokens to a coarse branch
4. Send residual high-frequency tokens to a detail branch
5. Use a lightweight gate to mix the two outputs

This is not a literal Fourier transform module.
It is a local frequency proxy designed to capture the same intuition at much lower engineering risk.

## Why This Version First

This version is a good first step because:

- it has a clear reference-backed story
- it keeps parameter growth controlled
- it reuses the existing "branch specialization" design language already present in `nav_refine` and `time_moe`
- it avoids heavy FFT code in the hot training path

## What To Measure

Main questions:

- does `freq_split` improve FID over `swiglu`
- does it help low-NFE sampling look sharper without becoming unstable
- does the gate actually use both branches rather than collapsing to one path

Useful diagnostics:

- `gate_mean`
- `low_energy_mean`
- `high_energy_mean`
- whether the detail branch becomes more active later in training or at later diffusion times

## Current Scope

This branch should be treated as a structural probe, not a final model family.

If the signal is positive, later versions can become more explicit:

- time-aware frequency routing
- multi-expert frequency routing instead of two-way mixing
- more faithful spectral features or wavelet-like decompositions
