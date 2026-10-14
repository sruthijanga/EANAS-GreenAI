# Research implementation boundaries

Based on Sections 4.1–4.8 and 5 of the provided 60-page manuscript.

## Implemented

- Five searchable blocks; operators: 3x3 conv, depthwise separable, squeeze-excitation-style channel attention, average/max pooling and skip, including dimension-correcting projection where required.
- Arch-A5 reference: stem 32; depthwise32, depthwise64/2, channel-attention64, depthwise128/2, residual-conv128; global pooling + task head.
- Controller: two-layer LSTM 128, categorical policy, REINFORCE with EMA baseline (Adam LR 0.00035).
- EA: population 20, tournament 3, mutation .2, crossover .8, elitist survival via best-scored population.
- Candidate training 10 epochs, SGD 0.025, momentum .9, decay .0005, cosine; final 200 CIFAR epochs or 100 ImageNet epochs, SGD .1. Five seeds documented.
- Objective: 0.50 classification error + 0.25 normalized latency + 0.25 normalized energy; hard feasibility; exact non-dominated sorting.
- Measured timing, synchronized power CSV integration, dynamic-energy idle subtraction, report provenance; measured power must be continuously collected during each profiling session.

## Limitations / unresolved manuscript information

1. An instrument-specific continuous power logger, synchronization and hardware run-time settings are inherently device-specific. A CSV adapter accepts already-collected measurements; it is not an automated Raspberry Pi/Jetson acquisition driver.
2. Power sampled at 100 Hz may miss individual 10–30 ms inference windows. Energy is integrated across the full 1000-inference batch; use timestamp-compatible instrument samples, ensuring sufficient coverage. Dynamic power is clipped at zero after idle subtraction.
3. Provided normalized reference values of 25 ms and 1 J are placeholders chosen for a reproducible default; record validated pre-search calibration values before publishing claims.
4. The paper does not specify a unique DAG topology encoding, SE attention internals or residual block exact structure; the code fixes a five-stage serial DAG and shape-compatible skips. This is one defensible implementation, not proof of identity to an unavailable original.
5. NASNet-Edge and DARTS-Compact cannot be recreated faithfully without genotypes and architecture references; no fabricated proxy baseline is inserted.
6. MLPerf Tiny workloads require task-specific models and runtimes; the official task launcher is provided, not undocumented task emulation.
7. A dynamic power surrogate is estimated, not physically measured; results are marked `estimated_calibrated_power`.
8. All results must come from actual saved logs; manuscript tables and figure measurements are not inserted as synthetic data.
