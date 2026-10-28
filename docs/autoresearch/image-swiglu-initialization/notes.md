# Evidence and implementation notes

The response-geometry result7051da1 fails the predeclared common-direction
priority rule. It does not justify adding a PSD or monotonicity restriction.
Its own-space negative witnesses remain valid within their measured directions.

The completed cold initialization control gives matched SiLU64 FID178.056840,
versus default SiLU64 190.755168, on the fixed20k endpoint. Native K1 is
173.436914 and default Swi86 is195.310097. The source is the sealed
oct28-image-clock-training-v2/metric-output/result.json. This motivates one
missing SwiGLU initialization control before another ODE-only design.

The new Swi86 has exactly the same initial function and input derivative as
matched SiLU64 in real arithmetic, but additional parameter tangent directions.
Do not describe it as matching the NTK, dynamics or optimization trajectory.
Formal training rebuilds the original cold shared body; active inherited
weights are used only to expose shared-gradient checks in the preflight.

Implementation reuses immutable native draw/update/checkpoint functions and
the sealed cold-control training loop. In-memory callbacks supply only this
arm's constructor, matrix recorder and integrity verifier. No frozen source
file is edited. Read-only audits use independently spelled update/Heun/FID
helpers from the completed cold study, with separate output bindings.

Author witness72467/fa0044 exited0: initial float32 FFN difference0,
input/main-route derivatives and43shared active gradient tensors match;
all three extra gate/value paths open after one active update. This is a
readiness result, not a new training trajectory or quality measurement.

Fresh documented native33724/d2619a and metric53767/4cd010 invocations each
exit0, exactly once without mismatches. The parent binds both actual receipts
and preflight hashes. Controller61348 starts PID81673 from cold state; the
first observed100updates use98.6%of one CPU core. Six automatic phases remain
in progress; no new quality outcome is available.

Earlier README shorthand "fixed P" meant a fixed0.6clock with trainable P.
The root README now states this explicitly; frozen historical source remains
unchanged.
