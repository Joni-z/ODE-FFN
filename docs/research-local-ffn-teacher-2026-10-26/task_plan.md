# Task plan: Locate the fitting gap at the FFN boundary

## Goal
Determine whether the original ODEFFN already loses accuracy when fitting a fixed individual FFN function on a common input distribution, before attributing the whole-network gap to feature coordination.

## Progress
- [x] Revalidate the prior literature turn as progress: a source-verified review and mathematical scope clarification were saved.
- [x] Inspect the existing teacher experiment, original constructors, controls, and current resources. No training process is live; Lenovo SSH times out.
- [x] Register the local teacher functions, common inputs, eighteen cold cells, and interpretation rules.
- [x] Implement and preflight capture, native constructors, exact representers, analytic derivatives, and resume.
- [x] Freeze sources and launch the monitored two-worker CPU study once.
- [x] Independently audit native outputs, consumed streams, checkpoints, derivatives, costs, and terminal processes.
- [x] Plot every result and report what this changes about the next design question.

## Constraints
Training seed 0 only. No architecture, activation, head, loss penalty, learning rate, width, or horizon sweep. Preserve all prior artifacts, submitted Report1, and dirty worktree. This is a localization experiment, not an architectural nomination or a teacher-based generative method. No commit or push.

## Resources
Mac mini: 4.69 GiB available RAM, about 85 GiB disk available at the initial check. Use at most two one-thread CPU workers; no GPU reservation. Lenovo timed out on a real SSH attempt this turn and is not assigned work. Never restart on an observation timeout.

## Errors and observations
The first discovery batch produced excessive output; later reads are bounded. Queries found no prior single-FFN teacher-fitting experiment in the project Markdown records. Missing new study paths were expected, confirming the new output root is unused.

## Status
Complete local study, analysis and mathematical follow-up. All eighteen cells completed 5000 updates; supervisor session 99697/PID 2179 exited 0 after 72.09 seconds. Independent audit session 23987/PID 2289 exited 0 in 10.56 seconds. All twenty recorded supervisor/worker/audit PIDs are absent. Three figures were visually inspected. The six target functions show opposite own-family fitting preferences; the ODE deficit on SwiGLU targets is already local, with 81.8%–90.8% of excess outside the empirical affine component. A separate QR projection audit and 36 origin-Hessian identity checks pass. This does not nominate an architecture, resolve expressivity versus optimization, or complete the original goal.
