# Directional dominance needs an equal-dimension control

The completed interventions identify which hidden correction carries the
observed output and sampling change. A large tangential fraction alone does
not show that this is an unusual property of an ODE-inspired layer: the
potential normal has one dimension and its complement has63.

## Exact geometric reference

For a fixed feature difference delta in d dimensions and an independent
uniform unit vector r, rotational symmetry and trace give

```text
E[r r^T] = I/d
E[||r r^T delta||^2] = ||delta||^2/d.
```

In d=64, a generic one-dimensional normal captures1/64of squared feature
error on average. Averaging projections onto every vector in any complete
orthonormal frame gives exactly the same value, without a stochastic estimate.
A deterministic normalized Hadamard frame verifies that identity on every
one of the2880retained states. This reference is geometric, not a hypothesis
that trained errors are random or isotropic.

## Existing observations after accounting for dimension

| Task | Model | Normal error share | Relative to 1/64 |
|---|---|---:|---:|
| Ring8 | Shared P | 0.258 | 16.489 |
| Ring8 | Palindrome | 0.109 | 6.954 |
| Ring8 | Alternating | 0.168 | 10.749 |
| Product16 | Shared P | 0.047 | 3.038 |
| Product16 | Palindrome | 0.063 | 4.023 |
| Product16 | Alternating | 0.034 | 2.158 |

The normal component is smaller in absolute error energy but enriched by
2.158--16.489times relative to the geometric reference. Thus
"energy-only is insufficient" does not mean "the potential normal is irrelevant."
The current directional effect must be compared with a dimension-matched null
before claiming a special directional sensitivity or ODE-specific mechanism.
The frame identity error is4.337e-19; all six published normal shares
reconstruct independently. [Exact audit](direction-dimension-audit.json).

## Next falsifiable control

Keep every checkpoint, all80inputs, all six separate layers and the continuous
reference. Bind one data-independent orthonormal frame per case/layer using
evaluation RNG0, with all64frame directions retained. For each unit direction r,
replace the native local output by native-rr^T delta and by reference+rr^T delta.
These remove a one-dimensional component and its63-dimensional complement,
exactly matching the dimensions of the potential-normal/tangential split.
Compare their mean full-output reproduction and squared output response with
the actual potential split, retaining per-bank/layer results. The complete
frame has exact mean feature-removal energy||delta||^2/64, so dimension cannot
explain a remaining response difference by itself.

The strong directional-specificity prediction is F_tangential greater than
the frame-mean complement F in every model. A nonpositive paired difference
rejects that six-case claim. Separately test whether F_normal exceeds the
frame-mean one-dimensional F; these are different hypotheses and may have
opposite outcomes. Keep every layer, direction and bank. Input-bank intervals
describe fixed-weight variation; no training replicate is added.

This is the next mechanism experiment, not an extra architecture or a new
response-function search. It has not run. No larger sampling comparison or
new image candidate is selected from the small-source pilot alone.
