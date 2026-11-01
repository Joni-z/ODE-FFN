# Synthetic audit evidence notes

## Verified inputs and replay
- Frozen original study: `/Users/mr.z/ode-workspace-prep/architecture-search/cubic_split_20261026`.
- Frozen transfer study: `/Users/mr.z/ode-workspace-prep/architecture-search/joint_dependence_20261026`.
- Frozen shared backbone: `architecture-search/conservative_readout_20261026/source-v2`.
- Rechecked 688 recorded file bindings across the two study manifests; all match.
- Twenty saved 5k checkpoints reproduce their full saved field predictions exactly on CPU.
- Three Product16 sampling runs reproduce saved samples exactly with the recorded 32-step Heun sampler.
- Independently recomputed saved-sample SW2 for all twenty endpoints and checked all fifteen Report2 numeric rows.
- No optimizer updates, training, sampling selection, or report compilation was performed.

## Configuration
- Time-conditioned vector MLP, width64, three residual blocks, LayerNorm/time FiLM, residual scale1/sqrt(3), direct velocity output, no attention.
- Professor ODE K2 is a real control; the positive FreeISRU arm is an ordinary free-matrix activation FFN.
- Exact positive ISRU runtime provenance records CPU. Later image experiments used MPS, separately.
- Initial experiment used Gaussian2D/Ring8/TwoMoons; later ISRU screen used Gaussian2D/Ring8/Product16. These should not be conflated.

## Limits to preserve
- Shared backbone initialization and data streams, near-parameter Swi64 and larger Swi160 are meaningful controls.
- Special ISRU matrix initialization versus standard SwiGLU prevents activation-only attribution.
- Fixed5k budget is not convergence. Product16 ISRU/Swi160 field errors continue dropping substantially between2.5k and5k.
- Product16 field advantage does not give a comparable sample SW2 advantage. Dependent-sign field ranking reverses.
- The tied cubic candidate is different from FreeISRU; its reported Gaussian error exceeds the original ODE.
- Existing Report2 .tex/.pdf remain unchanged; phrase at report2.tex:38 needs target qualification in a later report revision.

## Verification command
```sh
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B docs/synthetic-audit-2026-11-01/verify_existing.py --workspace /Users/mr.z/ode-workspace-prep --repository /Users/mr.z/ODE-FFN-upstream --output /Users/mr.z/ODE-FFN-upstream/docs/synthetic-audit-2026-11-01/verification.json
```

## Fresh review outcome
The reviewer returned WARN: no fabricated ground truth or performance normalization was found in the audited path. The exact field results are supported within their fixed budget. General ODE/Transformer, converged, and activation-only claims are unsupported. Private full transcript and route are retained under `.aris/traces/experiment-audit/2026-11-01_run01/`; the published audit records this as provisional same-family review.
