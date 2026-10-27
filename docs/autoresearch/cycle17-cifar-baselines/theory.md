# What this image comparison actually tests

The professor's map is F_K(h)=sum_(j=0)^K s(h)^j P^j h/j!, with a scalar
input clock selected once. Its complete derivative is

    J_F = T_K(sP) + [P*T_(K-1)(sP)*h] outer grad(s).

Both input dependence and parameter reuse are preserved in the bridge. This
is a frozen-duration linear-flow polynomial, not a guarantee that the whole
Transformer is an ODE solver or that a better internal solve improves FM.
The K1 control distinguishes higher-order computation from the basic gated
feature response; Swi86/129 distinguish parameters from executed matrix work.

For images x and noise e, x_t=t*x+(1-t)*e, the capped targets and predictions
are v=(x-x_t)/delta and v_hat=(D-x_t)/delta. Therefore

    ||v_hat-v||^2 = ||D-x||^2/delta^2.

This preserves a JiT-style clean prediction trained in velocity-loss space.
The conditional optimum is a clean conditional mean; the image pilot has no
analytic oracle field to measure that optimum directly. At the final patch
head48outputs can be spanned by64hidden coordinates. The earlier vector
benchmark's dimension-forced affine null space does not follow here.

## Exact low-rank computation of empirical FID

Let feature matrices X,Y have n,mrows and dcolumns. Center them to Xc,Yc;
use unbiased sample covariances Cx=Xc^T Xc/(n-1),Cy=Yc^T Yc/(m-1).
Write A=Xc/sqrt(n-1),B=Yc/sqrt(m-1). The nonzero singular values of A B^T
have squares equal to the nonzero eigenvalues of Cx*Cy. Consequently

    tr sqrt(Cx^(1/2)*Cy*Cx^(1/2)) = ||A B^T||_*
    FID = ||mean(X)-mean(Y)||^2 + ||A||_F^2 + ||B||_F^2 - 2*||A B^T||_*.

This avoids an unnecessary2048Dmatrix square root when sample covariances are
singular. It is the same empirical-covariance quantity, not a new metric or a
bias correction. Use float64 SVD and only clamp a negative roundoff result
whose magnitude is below1e-6; larger negatives are errors. Small-sample FID
bias remains. A finite low score is not a proof of mode coverage or a valid
comparison with FID-50K.

Feature preprocessing/weights are those of the pinned, unmodified
[torch-fidelity extractor](https://github.com/toshas/torch-fidelity/tree/v0.3.0).
Its [FID implementation](https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_fid.py)
is the independent covariance-formula fixture reference. Our low-rank identity
is verified numerically before trained outputs are observed.
