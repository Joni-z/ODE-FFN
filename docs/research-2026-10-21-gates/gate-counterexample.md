# Perfectly conditioned P does not bound the gated layer's Jacobian

This is an exact construction in the professor's K=2 architecture, not a trained
model result. It isolates the input gate without any large matrix powers.

Choose P=I in two dimensions. Both eigenvalues and both singular values of P
are exactly 1. Let g(h)=h1, which the prescribed two-layer SiLU gate realizes
exactly because SiLU(z)-SiLU(-z)=z. Use first-layer rows (1,0), (-1,0),
second-layer weights (1,-1), and zero biases.

Keep the original s(h)=0.2+0.8*sigmoid(g(h)/10).
At h=(0,R), s=0.6 and grad(s)=(0.02,0). The K=2 layer becomes

    F(h) = [1+s(h)+s(h)^2/2] h

Its Jacobian at those inputs is exactly

    J(0,R) = [ 1.78       0  ]
             [ 0.032 R  1.78]

Both Jacobian eigenvalues remain 1.78 for every R, but its largest singular
value is

    sigma_max(J) = [sqrt((0.032 R)^2 + 4*1.78^2) + |0.032 R|] / 2

It grows without bound as |R| increases. Bounding the eigenvalues of P,
bounding its largest singular value, or making all its singular values equal
does not eliminate this source of sensitivity.

| R | Largest singular value of J |
|---:|---:|
| 1 | 1.796072 |
| 10 | 1.947177 |
| 100 | 3.993408 |
| 1,000 | 32.098708 |
| 10,000 | 320.009901 |

Autograd in float64 agrees with the displayed Jacobian to 1e-12 relative/absolute
tolerance at all five inputs. The numeric records are in
[gate-counterexample.json](gate-counterexample.json).

Reproduce with:

```sh
python scripts/check_gate_counterexample.py \
  --output docs/research-2026-10-21-gates/gate-counterexample.json
```

## Scope and implication

This disproves a global bound for the standalone ODELayer over unrestricted h.
It does not show that our trained velocity network explodes: the surrounding
LayerNorm/FiLM restricts the hidden inputs for fixed finite network weights and
t in [0,1], and the observed input domain can be much smaller.

The useful experimental consequence is to record hidden-input magnitude and
the gate derivative along with P. A bound involving P, a bounded input domain,
and the gate-network derivative is possible; a bound based only on P misses a
term. The present gate-removal experiment tests whether that term buys useful
sample quality before we design a constraint for it.

The same problem survives an exact exponential solve. For P=I,
F(h)=exp(s(h))h, and its Jacobian at (0,R) is
exp(0.6) times the matrix with rows (1,0), (0.02R,1).
Thus this mechanism is distinct from Taylor truncation error.

## A bound that includes the missing quantities

Suppose ||h|| <= R, ||P|| <= M, s <= s_max and ||grad(s)|| <= L_s.
Define the scalar polynomial t_K(z)=sum from k=0 to K of z^k/k!.
The triangle inequality and submultiplicativity give

    ||J_K(h)|| <= t_K(s_max M) + M t_(K-1)(s_max M) R L_s

For the original gate, L_s <= 0.02 L_g whenever g is L_g-Lipschitz.
This bound is conservative, but it separates the four quantities we need to
monitor: matrix norm, duration, hidden-input magnitude and gate derivative.
Any strictly positive cap on M alone is insufficient while R and L_s remain
uncontrolled. The counterexample achieves growth through R with the other
quantities fixed.
