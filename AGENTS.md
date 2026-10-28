# Project working conventions

- The canonical repository is https://github.com/Joni-z/ODE-FFN, branch `main`. The user reads code on GitHub. Push completed, verified changes promptly rather than accumulating local-only work. Normal pushes are authorized; do not force-push published history.
- Use **Zhizhe Zhang** for Git author and committer names. **Jonathan Zhang** is also accepted. Do not create commits under a Chinese name. The repository's GitHub-associated email is `195848701+Joni-z@users.noreply.github.com`.
- Keep the README and research reports in English. Explain results to the user in concise Chinese unless asked otherwise.
- Report1 is final at commit `994e660921d2885592704f5c471bd43a12b1c6ef`. Do not edit, rebuild or replace anything under `reports/report1/`; historical draft wording there does not change its final status. All work after that revision belongs to Report2. Maintain `reports/report2/` separately and update its evidence index as research progresses.
- Maintain `task_plan.md` and `notes.md` when priorities or findings change. Read their current-priority sections before selecting work. Link the canonical follow-up note, retaining positive results, failed transfers, unresolved validity gaps and the next concrete action.
- Formal training uses model seed **0** only. Do not add multi-seed experiments. Preserve registered historical data streams and state their initialization separately.
- Compare ODE candidates with strong ordinary controls, including SwiGLU and relevant parameter/compute controls. A gain over an old ODE alone is insufficient. Record learned P trajectories, not just final generative metrics.
- State the hypothesis, exact operator, matched controls, budget and stopping criterion before a new experimental family. Avoid bundling tricks or reviving failed settings with unregistered rescues.
- Preserve frozen source snapshots, raw runs, checkpoints and the submitted Report1. Keep datasets, environments, checkpoints and raw execution logs outside Git; publish code, configurations and curated evidence.
- Monitor running studies. Re-poll an existing live process after an observation timeout; do not restart merely because the observation expired.
- Run new experiments on **Mac mini and Lenovo**. Do not depend on MBP or torch for experiment execution. Configure GitHub publication on Mac mini directly rather than asking the user to restore another machine solely to push commits.
- Sync reachable machines through GitHub. Check for local changes before updating a checkout, and use fast-forward updates. Never discard another machine's work to make a pull succeed.
