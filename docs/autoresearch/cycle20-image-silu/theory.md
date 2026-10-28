# The ordinary function class behind constant clocks

For scalar fixed c, one professor solve is a fixed matrix
`M=T_K(cP)`. Two solves aroundSiLU therefore compute `M2 SiLU(M1 h)`.
An ordinary width64bias-free two-matrixSiLU block can represent each such
frozen function by setting W1=M1,W2=M2. This is a real-arithmetic containment
statement, not a theorem about default initialization or Adam trajectories.
For K2, arbitrary matrices need not be representable by the polynomial map;
the ordinary class can be larger. Existing local teacher experiments do not
justify an optimization rescue or an image-generation conclusion.

For a general widthm block, `F(h)=W2 phi(W1h)`, withphi=SiLU:

    J_F v = W2 [phi'(W1h) * (W1v)]
    phi'(u) = sigmoid(u) + u sigmoid(u)(1-sigmoid(u)).

Two bias-free matrices contain128mweights and execute128mdenseforwardMACs
per token. Minimum integer width covering16642parameters is131; matching
K2's24704matrixMACs gives193. Width64preserves the frozen core shape. These
three roles determine the cohort before training.

The common backbone, residual paths and time conditioning are unchanged.
No invertibility, convex-gradient, continuous-depth, higher-order accuracy or
full-velocity stability property follows from this ordinary local form.
The experiment asks whether ordinary training is competitive at the stated
budget; it cannot prove which mathematical mechanism caused any difference.
