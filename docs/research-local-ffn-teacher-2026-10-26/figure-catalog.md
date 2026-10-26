# Figure catalog

All figures are generated from audited run arrays and exported tables with standard Matplotlib. PNG and PDF versions are saved; all three PNGs were visually inspected and are readable. There are no error bars, invented replicates, smoothing or selected subsets.

## local-fitting.png / local-fitting.pdf
Purpose: show all eighteen endpoint risks and locate their affine/nonaffine components. Each panel fixes teacher family and position; bars show all three students. Stacks use the same empirical projection basis and add to relative teacher MSE. Caption must state squared-error normalization, common inputs, 5000 updates, and the post-result descriptive nature of decomposition. Notice the opposite own-family preferences and the remaining component of the ODE deficit on SwiGLU targets. This places a gap within an isolated module, without proving a class lower bound. Check that the tiny own-family ODE bars retain explicit numeric labels; do not compare panel heights as equal-difficulty targets. Axes vary by panel and begin at zero.

## learning-curves.png / learning-curves.pdf
Purpose: show every saved trajectory rather than only endpoint ordering. Each panel fixes teacher/position; colors identify all three student families. Values are the four actual saved validation risks, connected by straight segments on a logarithmic y axis. They do not imply measurements at intermediate coordinates. Notice the initial proximity and fast ODE self-fit, and the remaining cross-family gaps at the fixed budget. Starting advantages and optimization remain alternative explanations; these curves do not justify an unregistered longer run.

## P-trajectories.png / P-trajectories.pdf
Purpose: retain the behavior of every trained original P rather than interpreting endpoint risk alone. Six ODE cells each contribute two curves, with all four saved checkpoints. Notice more growth when fitting SwiGLU targets. Norm growth is an association, not a demonstrated cause or a full-network Jacobian certificate. This does not reopen rejected spectral-cap or clock settings. Axes vary by panel and do not begin at zero, to show the recorded changes explicitly.
