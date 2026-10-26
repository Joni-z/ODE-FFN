# Fixed-point versus task-error diagnostic

Registered after the negative Phase A screen and before reading the paired per-input outcomes. This is a read-only checkpoint diagnostic, not a new training branch or a way to nominate a failed candidate.

The saved aggregate ODE anchored probe residual drops from4.38882 at iteration3 to.0979079 at12, while its full validation field MSE rises from.470643 to.948070. Those aggregates concern different input banks. To check what happens on the same inputs, use exactly the existing256 probe inputs, all six trained endpoints, and iterations3/6/12. Evaluate the analytic population velocity on these already exposed probe inputs. Save each input's field squared error (coordinate mean) and hidden fixed-point residual RMS. For the accumulated control the same next-state change is logged, but is not called an implicit-equation residual.

Report all18 aggregate rows and the fraction of inputs on which residual decreases while field error increases from3 to12. Retain absolute values, input/checkpoint/source hashes and raw rows. No additional iteration counts, fresh test targets, model selection, generation or training are allowed in this diagnostic. Decreasing residual does not establish a global contraction; its value does not bound task error without additional assumptions.
