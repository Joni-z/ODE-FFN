# Pre-outcome resource and correctness check

Prototype only, before any quality outcomes. Candidate F(h)=J_L(h)^T SiLU(L(h)), where L is one unchanged professor scalar K2 layer. Compare exact same local map with its clock-adjoint term omitted, the original two independent solves, ordinary tied-ridge128/192, and ordinary SwiGLU128/192. These are seven predetermined settings.

Require exact VJP value agreement and mixed parameter/input derivatives with a separately autograd-differentiated original layer, a symmetric local input Jacobian for the full version, correct clock-term difference, no duplicated/dormant parameters, whole-model finite differences under a nonzero head, and unchanged common backbone/buffers/first encoder initialization.

For the native64-wide three-block Product16 model, execute actual matrix-operation counting for inference and one loss/backward, including every ordinary-tensor derivative feature. Record isolated one-thread CPU forward and loss/backward median group timings on a deterministic batch512, using a nonzero receiving head. No optimizer update or quality evaluation during profiling. Dense counts exclude pointwise work; timing is a microbenchmark, not full training throughput.

Resource comparability requires full candidate parameters at most80% of original, inference MACs at most110% of SwiGLU192, and loss/backward MACs at most110% of original. Timing is descriptive because tiny CPU kernels can vary; a large latency regression must be reported rather than hidden by dense counts. A pass only makes one separate cold protocol eligible for scientific review. No performance claim, architecture acceptance or automatic training follows from correctness/cost alone.

The common zero head makes initial complete outputs zero; hidden functions and output-head responses differ, so do not claim complete parameter-Jacobian pairing. Generic tied controls intentionally bracket the candidate's parameters and executed work; they need not be exactly matched. No slope, spectral, activation, head, objective or initializer sweep.
