# Fixed image baseline question

Does the professor's literal input-clock K2 block improve clean-image
prediction and generated-image quality against K1 and ordinary parameter/
work-covering SwiGLU in a small actual JiT backbone?

- Motivation: completed vector studies did not yield a useful successor;
  their global readout restriction differs from the per-patch image head.
- Hypothesis: a measurable image-setting interaction may exist. There is no
  theorem that internal Taylor order must improve the external FM objective.
- Controls: K1 shares initial P/controllers; Swi86 covers parameters and
  K1 matrix work; Swi129 covers K2 matrix work. Everything else is fixed.
- Evidence: cycle16 verified data, local clock semantics, active gradients,
  resume and throughput. No CIFAR quality outcome has been seen.
- Test: four cold5000-update trajectories, paired data streams, fixed final
  validation, Heun32generation and1024/2048InceptionFID.
- Stop: failure of the fixed joint fitting/generation guards stops this image
  setting. No old failed operator, hyperparameter sweep or best snapshot.
- Success boundary: even a positive original-K2baseline is not a novel ODE
  method or synthetic transfer result. The useful-successor goal remains open.
