# The exact constant-clock map and what changes

For one scalar-input-clock layer, write

    F_K(h) = T_K(s(h) P) h,
    T_K(A) = sum_(j=0)^K A^j/j!.

At a fixed scalar c, the complete layer is a fixed linear map

    M_K = I + c P                         (K1)
    M_K = I + c P + .5 c^2 P^2           (K2).

Thus a constant-clock FFN is exactly `M2 SiLU(M1 h)` in real arithmetic.
This does not equate floating-point evaluation orders. Formal predictions
use the literal recurrence; an independent double-precision folded map
checks the algebra through the entire unchanged image predictor.

The native derivative has an additional rank-one term:

    J_F = T_K(sP) + [P T_(K-1)(sP) h] outer grad(s).

Replacing s by c changes both the polynomial value and this derivative,
then changes all downstream inputs. Small variance of s on native states
therefore does not bound the complete intervention's effect. No local P
spectrum alone controls the remaining attention/readout/velocity response.

The calibration minimizes squared scalar-clock deviation on its 4,096native
token states. It does not minimize predictor error, trajectory error orFID.
The constants are fixed across external time and inputs. Their use can cause
distribution shift on generated states; a frozen necessity result is not a
theorem that a constant-clock model cannot be trained successfully.

Constant-clock folding identifies a two-matrix SiLU function, but says nothing
about which retrained matrix functions are reachable through a fixed-orderP
parameterization. The prior effective-matrix teacher study already rejects
a general optimization rescue. No such training or reparameterization is
selected here, and no useful new ODE successor is assumed.
