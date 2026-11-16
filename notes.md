# Current ODE mechanism evidence

Canonical directory: docs/research-assessments/2026-11-16-ode-mechanism/.
Image frozen-DTaylorK8and evolving-state synthetic ODEare different computations.

## What is established

Better internal ODEaccuracy alone does not improve generation. Hidden numerical
bias reaches the complete field. At fixed weights, midpoint inference can
improve generation even where its field MSEis worse. The measured full-field
perturbation predicts all8pooled target-effect directions without fitting.
Rare particle sensitivities do not imply population sampler fragility.

Population-resolution-results/: all4weights/2streams/8x4096banks/5meshes,
40rows/720metrics/32new contrasts; all4producer+4NumPyexits0. Metric error
<=1.12e-16; recursive exact statistics agree0. Sourcea525eb2/5127files unchanged.
At2048ProductRK4selection SW2.053->.049,confirmation.048->.045,all16banks
improve. Adjusted squared-loss intervals[-5.279e-4,-3.162e-4]and
[-3.396e-4,-1.915e-4]. All40pooled comparisons favor midpoint; Ring8bank
intervals cross0. Same-weight effect; ordinary architecture ranking requires
matched fresh banks. Both figures inspected, all original failures preserved.

## Analytic field prediction now complete

learned-defect-field-results/: four frozen models/two full4,096-input streams.
BOTH modified-field defects pass through all six local flows and unchanged
network. No fitted scale or observed field change enters the predictor.
ProductRK4 scores0.962/0.962,relativeRMS0.196,alignment0.987; all8case scores
0.957-0.983. All5time bins have positive scores; low-noise-time errors larger.
All8oracle-risk signs predicted. ProductRK4field risk increases+2.335e-5/
+2.601e-5(actual),+2.418e-5/+2.356e-5(predicted), while generation improves.

Producer52369actual0in186.177s; NumPy7963actual0,error2.17e-18. All refined
numerical cases qualify. Only1/8original64-reference cases qualifies on the
full sets; all old flags remain recorded. Source055a860/5175hashes unchanged,
GPU4released. Both actual figures inspected; original/layout-only analyses
90318/56920actual0and numeric summaries identical. Final reports unchanged.

The remaining ODE-specific link predicts target quality from this analytic
forcing through the ACTUAL learned sampler. Median eager256observer7.232s/
B4096projects primary full Heun2048transport to131.653GPUhours. Runtime
protocol fixes compilation of the same coupled RK4step before any such long
run; FP64,256steps,both defects,all banks and gates stay fixed. No target
transport from the analytic forcing has run yet.

## Runtime and audit

Mac warm04f11f75and145warm41450454unchanged. Only empty physical1454-7;
0-3prohibited, no foreign task touched. Population queue940668/tool29874terminal0;
collector86573terminal1from source-hash Path type, cached analysis41648and50553
terminal0. All failed logs/frozen sources retained, no sampling rerun. Resource
polls show mean utilization56.733-60.420%on4cards; all contexts now released.
Final reports97files/11bindings and five native image files unchanged.

Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
145: /data2/zz5070/ODE_FFN/population-resolution-20261117/.
Earlier journal: docs/history/2026-11-17-after-population-resolution/.
