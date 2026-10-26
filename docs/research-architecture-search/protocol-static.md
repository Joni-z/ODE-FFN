# Stage1: static polynomial versus collapsed matrices

Frozen before results. Data: original Gaussian anisotropic2D and Ring8. Seed0,5000updates,batch512,AdamW3e-4,wd0,clip1,uniformt,original Gaussian independent path. Width64,depth3. Fresh paired references in the same Windows environment; baseline SwiGLU rerun in this stage.

Three arms per target:

1. `ode_fixed`: professor's two K2 solves, duration.6, no input gate network.
2. `ffn_poly_init`: two bias-free64x64 matrices around SiLU, initialized as `B_i=I+.6P_i+.18P_i²` from the exact same initial P draws as arm1. All surrounding initial weights/RNG draws are preserved.
3. `swiglu`: existing practical baseline, nominal hidden128 with standard2/3 convention.

The first two arms have identical active parameter counts and identical initial hidden functions up to floating-point roundoff. The free-matrix arm uses fewer matrix multiplications; record this rather than call the compute equal. The K2 feasible matrix set differs globally; a training difference does not prove a pure optimizer cause. SwiGLU has a different parameter count, explicitly reported.

Primary screen: final validation oracle MSE and common interpolant field/Jacobian error. Validation SW2 and training wall time supplement the screen. Use no test set for architecture selection at this stage. Save all intermediate P/effective matrices or free-B snapshots, as well as common-state arrays. If both static arms remain substantially worse than SwiGLU and the free matrix matches or improves the polynomial, do not develop more static polynomial constraints by default.

This stage tests an unresolved mechanism; it is not sufficient by itself to declare a useful new architecture.
