# Upstream baseline notes

User instruction,1 November2026: copy standard repository code rather than
reimplementing a training framework. Keep the active repository simple; the
FFN should be the replaceable part. An executed kernel is not proof that a
bespoke benchmark has correct learning semantics.

Investigate both official JiT and official flow-matching image examples. A
public CIFAR result from a U-Net does not automatically validate a DiT/JiT
FFN comparison. Preserve that distinction when selecting the source.

## Selected source and scope
Official JiT cbc743a2, MIT, is copied to the root; Meta flow_matching11568d37
was reviewed but its published CIFAR example is a U-Net. It is not substituted
for the Transformer research question. Eleven upstream files were imported;
five remain byte-identical and six have listed adapters. The original SwiGLU,
all objective/sampling/EMA methods are AST-identical. Full differences are
in UPSTREAM.patch. Existing free-ISRU is the only candidate implementation.

The complete previous source is published on research-archive-20261101 at
2d4e5ec. Tracked old active src/configs/synthetic/scripts/tests and Slurm
entrypoints were removed only in the new worktree. Original checkout, raw
runs, reports and runtime source remain intact. Historical docs stay as
evidence and are never imported by the new entrypoint.

Author CLI7626/9650c9 and compatibility99495/b7ab70 pass. Fresh documented
CLI588a70 and check5f8134 each run once, exit0, no discrepancy. Expected
inherited AMP FutureWarnings are retained. Actual default CPU outputs,
input/parameter gradients, epoch/Adam/two-EMA update and both samplers match
the placement-adapted author reference bitwise. Candidate gradients, stream
pairing, checkpoint schema and a tiny MPS update pass. This is not FID evidence.

## Setup issues retained
Pip Git transport failed earlyEOF (22115/302e8c). The identical author-fork
commit was installed from its pinned codeload archive (32632/7a8dd4,exit0).
No existing environment was modified. One combined delete/add README patch
was rejected for duplicate paths and was applied as two targeted operations.
No experiment source or output was lost.
