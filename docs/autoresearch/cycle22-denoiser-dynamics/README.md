# The dynamics of the supervised denoiser

Read the [mathematical object and proofs](derivation.md) and
[fixed numerical protocol](protocol.md). This checks objective identities;
it does not train or nominate an architecture. Review assurance is recorded
separately from numerical reproduction.

Use the existing check interpreter, with torch2.10.0 and numpy2.2.6. It does
not import image models or require einops. Prepare only a new runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python docs/autoresearch/cycle22-denoiser-dynamics/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle22
```

After registering the source and checking the proof scope, run once:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle22/source/controller.py
```

The controller records both child commands, PID/create-time, exit codes and
two-second CPU/RSS telemetry. Follow the same tool handle on observation
timeouts. A failed run is preserved, not overwritten. Three existing-law
validation cohorts and all probe derivatives remain outside Git; curate
only completed numerical evidence. Test data, training settings and Report1
remain unchanged. No GPU or remote host is needed.
