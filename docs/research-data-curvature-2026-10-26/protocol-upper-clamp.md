# Exploratory fixed-weight upper-clock intervention

Written after the derivative and clock-value summaries were viewed, before computing this intervention's function risks. The SwiGLU-target students have durations in roughly [0.86,1], with 54.375% of the sampled durations above 0.99. Local curvature agreement does not establish that one globally fixed clock approximates the complete function.

For each of the six original-ODE students and three original-ODE teacher modules, evaluate all 20000 existing common validation feature inputs. Compare the unchanged native module with a separate clone whose two durations are fixed to the existing upper endpoint 1.0 everywhere. Use the native provided-duration forward path; preserve every parameter. Do not select a clock value, optimize a constant or retrain anything. Check the constant-map implementation independently as Q2(1) SiLU(Q1(1) h) in float64.

Report normalized native-versus-clamped function difference for all nine modules. For every student report native and clamped risks against its unchanged teacher target, with absolute and relative errors. Native float32 outputs must replay the sealed validation arrays bitwise at the historical batching. Preserve clamped arrays and source. This is an exploratory fixed-weight intervention, not a new trained architecture, an optimizer diagnosis or a proposed generative method.

If clamping strongly alters the fitted function, do not infer a globally constant-clock description from small local curvature differences. Small function changes would justify a paired constant-clock core experiment as a separate mechanistic question; they would not identify why the original clocks saturated or imply that removing them improves FM training.
