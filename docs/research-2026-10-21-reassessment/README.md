# ODE-FFN research reassessment

This review pauses new training and asks what the existing results actually imply about the original ODELayer. Submitted Report 1 is unchanged.

- [Main assessment in Chinese](assessment.md): the research question, priorities and reasons to stop or revise a direction.
- [Evidence audit](evidence.md): all archived synthetic experiment families, current image-code distinctions and explicit access limits.
- [Mathematical derivations](mathematics.md): frozen versus state-dependent dynamics, matrix parameterization, gate derivatives, exact FM target geometry and the JiT connection.
- [Focused primary-source reading list](literature.md): relevance, publication status and the sections actually reviewed.
- [Mathematical verification](math-checks.json) and [reproduction script](verify_math.py): deterministic CPU checks, not new training experiments.
- [Working record and workflow attribution](notes.md), [completed plan](task_plan.md).

Run the checks from the repository root with PyTorch and NumPy available:

```bash
python docs/research-2026-10-21-reassessment/verify_math.py
```

The recommended next step is to settle the mechanism and its diagnostic predictions before choosing another architecture or launching another training grid. No proposed new training configuration has been implemented or submitted.
