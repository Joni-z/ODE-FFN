# Independent implementation review: state-clock comparison

**PASS: no blocking implementation issue found in the reviewed revision.** This is a correctness review of the registered comparison, not evidence that midpoint improves training. No source changes or formal training were performed during this review.

The helper's cached `Ph + P*first/2` equals `P*h_mid`; its frozen expression plus clock correction equals explicit midpoint. The tests compare values and every input/parameter gradient against a separately written two-field expression with an active controller. Neither clock evaluation is detached. The analytic clock JVP and full nonlinear input Jacobian include controller derivatives; the local-error reference integrates the actual state-dependent field with DOP853.

The wrapper retains two independent original P/controller pairs and middle SiLU. It introduces no trainable parameters or RNG draws beyond the original construction, and zeroes only the declared final controller affines. Frozen/Euler legacy compatibility is checked after applying that same initialization. Frozen/midpoint initial functions pair, but their last-controller weight tangents differ with a nonzero receiving head. Tests correctly restrict the formal zero-head full-model gradient pairing to frozen/midpoint, not Euler.

Functional P applications are counted manually; controller Linear hooks count both midpoint calls. Independent instrumentation of actual `F.linear` execution agrees with the MAC counter. On the 16D full backbone, Euler/frozen/midpoint use 81,984/106,560/131,136 dense MACs and each has 83,664 parameters. Reusing parameters therefore does not hide repeated computation.

The standalone class is absent from `ode_layers()`, preventing legacy one-clock-per-layer capture from accidentally overwriting its two stages. The registered runner evaluates native full-field Jacobians through `check_vector_field.py`; this review does not certify the older gate-attribution routine for the new family. Dedicated collection logs P separately and marks midpoint clock probes as counterfactual for Euler/frozen. It does not present those probes as executed stages or full-network stability certificates.

The protocol and three plans agree on six arms, three datasets, seed0, CPU/two workers, fixed training settings, and hash-bound Stage15 backbone initialization. The fresh constant-clock controls and the near-compute-matched SwiGLU256 are explicit. The protocol preserves the stopped-family boundaries and does not claim FM-specific benefits.

Independent command:

```text
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -m unittest tests.test_state_clock tests.test_state_clock_integration -v
Ran 12 tests in 0.310s — OK
```

Full trainer/resume, baseline trajectory reproduction and frozen-runtime audit remain separate preflight evidence owned by the main task; they are not inferred from these unit tests.

## Reviewed SHA256 identities

```text
131a28bcb21dd3fb82ea33828a31aa0f2ce418044b32356d54e3862833098371  synthetic/state_clock.py
ad614b795c1431cdd5df892d9754394eb4b9a72616f82726d6d6ea90704ecbaa  synthetic/models.py
11ecc22b36132680780b52cdfa42e18f8cfb57e60d06a3c47a112ae1a13762c6  synthetic/diagnostics.py
681b42c0ba6c10154e02643b432d284a014c4c36ee1a4a40fc7cf1e6557fc9a9  synthetic/train.py
57f40bbf6bb91687f41116348c0ba3018ad1e8ea04c33ff0e2a38e8ea37da8d5  scripts/run_architecture_study.py
c39ef01466da0b1b551fe5a7c6f35f53e1070ddd5075c01c4f2a1dcb29533346  tests/test_state_clock.py
8f99f693e6e837734fe55eeca9de522dcb3a90447472d1deb8a2ffe715e346e5  tests/test_state_clock_integration.py
b0686a449fd9fbff1ec7831390944e3dfde880f2ad3d0ed658a69755747f9f8f  requirements-synthetic.txt
af2105c01e3f294198280253621c9659a09533e4a44500d8bbc77a188c720a06  docs/research-architecture-search/protocol-state-clock.md
7ad2307beafb0dd5c1e2b7a2957ad0d4b0021843e8b16158df4f2efc900c2766  configs/synthetic/architecture_search/state_clock_v1_20261025_m1.json
33d5ece17650f7d36d92c5cde3b9ffb0883a64a2cfc01288dc4ebee8ae3ddf49  configs/synthetic/architecture_search/state_clock_product16_20261025_m1.json
76bdb083e016fc37dc0cdba992e1217493f27d524ecef4ceb3b302f3f9dd5dce  configs/synthetic/architecture_search/state_clock_queue_20261025_m1.json
```
