"""Full-backbone head-response checks with an explicitly active output head."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch


def verify(root: Path,native,core) -> dict:
    with np.load(root/'banks/thin/probe.npz') as data:x=torch.from_numpy(data['x'][:16]).double();t=torch.from_numpy(data['t'][:16]).double()
    rows=[]
    for family in ('ode_k2','swiglu128','swiglu192'):
        raw=native.old_build(128,family,'direct').raw.double()
        with torch.no_grad():raw.output.weight.copy_(.01*torch.sin(torch.arange(raw.output.weight.numel(),dtype=torch.float64)).reshape_as(raw.output.weight))
        outputs={};tangents={}
        for head in ('clean','shifted','direct'):
            model=core.CappedHead(copy.deepcopy(raw),head)
            weight=model.raw.input.weight;direction=.001*torch.cos(torch.arange(weight.numel(),dtype=torch.float64)).reshape_as(weight)
            value,tangent=torch.autograd.functional.jvp(lambda w:torch.func.functional_call(model,{'raw.input.weight':w},(x,t)),weight,direction)
            outputs[head]=value;tangents[head]=tangent
        denominator=(1-t).clamp_min(.05)
        torch.testing.assert_close(outputs['clean'],(outputs['direct']-x)/denominator,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(outputs['shifted'],outputs['direct']-x/denominator,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(tangents['clean'],tangents['direct']/denominator,rtol=1e-10,atol=1e-11)
        torch.testing.assert_close(tangents['shifted'],tangents['direct'],rtol=1e-10,atol=1e-11)
        magnitude=float(tangents['direct'].square().mean().sqrt());assert magnitude>1e-10
        rows.append(dict(family=family,probe_anchors=16,nonzero_raw_tangent_rms=magnitude,
            clean_scaling_max_difference=float((tangents['clean']-tangents['direct']/denominator).abs().max()),
            shifted_max_difference=float((tangents['shifted']-tangents['direct']).abs().max())))
    return dict(status='PASS',scope='post-training audit of native cold backbones with a fixed nonzero probe head; no optimizer update',families=rows)
