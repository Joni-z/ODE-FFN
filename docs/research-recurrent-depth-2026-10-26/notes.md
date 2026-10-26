# Research and execution notes

Date:2026-10-26, Asia/Shanghai.

- The prediction-space bridge completed24cells and failed its ODE-versus-generic screen. Its source/results remain frozen and are not extended.
- The new literature review identifies a different axis: recurrent computation inside one forward, plus supervision of intermediate predictions. LiFT and Looped-DiT provide relevant architecture/ablation precedents, not evidence for the professor block itself.
- Mac mini is idle for these studies; torch2.10.0, about4.84GiB memory available at preparation.
- Initial `ssh lenovo` timed out on port22. An asynchronous reconnection question is pending. Local preparation continues.
- The new experiment must separate weight sharing, intermediate supervision, ordinary extra capacity and initialization. It must include the same recurrence with SwiGLU.
