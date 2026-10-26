# Figure catalog

Three PNG/PDF pairs were generated from audited tables and visually inspected. All panels are readable. No error bars, smoothed trajectories or invented replicates are used.

## paired-fitting
Purpose: compare all six targets and all three students, with exactly paired initial P/W functions. Bars show relative squared errors on all 20000 validation inputs. Numeric labels preserve tiny control errors. Axes start at zero and vary by panel. Caption must distinguish the fifteen new cells from three historical references and state that ordinary SwiGLU has more parameters. Notice P's advantage on known-representer targets and the large shared gap on SwiGLU targets. This weakens a general P-coordinate explanation without proving universality.

## learning-curves
Purpose: retain every saved trajectory and show the P/W initial overlap. Colors identify all students; points are actual validation risks at 0/1000/2500/5000 updates, connected by straight segments on log axes. Reused controls are historical trajectories, not new training. Notice both cores accurately fitting fixed-flow targets while retaining substantial SwiGLU-target errors. The graph does not authorize a longer or retuned rescue run.

## P-trajectories
Purpose: retain all twelve raw P histories alongside the effective-map diagnostics in the tables. Each panel fixes target/position and shows both P matrices at every snapshot. Axes vary and do not begin at zero. Larger norms on the SwiGLU targets are descriptive; they do not establish a spectral cause. Effective-map and determinant records remain in the complete CSVs even though this figure focuses on P as requested.
