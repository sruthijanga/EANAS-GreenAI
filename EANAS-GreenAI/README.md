# EANAS-GreenAI

Reproducible implementation of the manuscript **Green AI: Energy-Aware Neural Architecture Search for Edge-AI Deployments**. Implements five-block CNN search, RL LSTM/REINFORCE, evolutionary mutation/crossover/tournament selection, normalized accuracy–latency–energy reward, Pareto selection, retraining, benchmark scripts, profiling and statistics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Use Python 3.10, PyTorch 2.2 and torchvision 0.17 to match the paper where supported. CUDA installation depends on the GPU driver and available PyTorch wheels. The packages are not bundled. Execution on newer PyTorch may work, but exact experimental reproducibility requires matching software and hardware.

## Fast functional verification (without downloads)

```bash
python scripts/run_search.py --smoke --synthetic --energy-mode calibrated --calibrated-dynamic-power-w 2.0
python scripts/generate_results.py --result-dir results/smoke
python scripts/train_final.py --synthetic --epochs 1 --max-batches 1 --workers 0
python scripts/profile_device.py --warmup 2 --runs 10
```

**Synthetic data and calibrated energy are exclusively for software tests. Never report these figures as empirical findings from the manuscript.**

## Actual CIFAR-10 architecture search

```bash
# Requires a synchronized measurement stream from a real instrument for measured energy.
python scripts/run_search.py --config configs/search.yaml --energy-mode measured --power-csv measurement.csv
```

`measurement.csv` requires columns `timestamp_s,power_w` (both numeric). The timestamps must use the same monotonic clock/time reference as the process and span the entire inference window. CSV capture must be provided by external measurement equipment, configured and synchronized by the operator. For an initial *surrogate* study, replace the last arguments with `--energy-mode calibrated --calibrated-dynamic-power-w 2.0`; **the chosen power must be measured/calibrated in reality**. This mode is an explicitly labelled approximation and cannot substantiate device energy claims.

Edit `configs/search.yaml` to set device, budgets, the `latency_ref_ms` and `energy_ref_j` references, and constraint values before searching. Default reference values are documented implementation choices, not values provided by the manuscript.

## Final training

```bash
python scripts/train_final.py --architecture arch_a5 --dataset cifar10 --data-root data --epochs 200 --device cuda
python scripts/train_final.py --architecture arch_a5 --dataset imagenet --data-root /path/to/imagenet --epochs 100 --device cuda
python scripts/run_baselines.py --models mobilenetv2 efficientnet_b0
python scripts/run_ablations.py --variant without_rl --synthetic --smoke --energy-mode calibrated --calibrated-dynamic-power-w 2.0
```

ImageNet root must contain `train/<class>/...` and `val/<class>/...`. CIFAR-10 uses 40,000 training/10,000 validation from official training samples, with the official test set held out. No automatic ImageNet redistribution is provided.

## Profiling

```bash
python scripts/profile_device.py --checkpoint results/final/cifar10_arch_a5_seed42/best.pt --device cpu --warmup 50 --runs 1000
```

For true energy, add `--power-csv measurement.csv --idle-w VALUE` and record instrument setup and power modes. The device profiling entry point measures processing latency, inference throughput and parameter storage size. Power integration is supported only when a synchronized stream is available. Without one, `energy_j` is null. An explicit calibrated-power mode is possible but clearly marked estimated. The profiler does not claim that the user has Raspberry Pi, Jetson or other edge hardware.

## Baseline boundaries

MobileNetV2 and EfficientNet-B0 load from torchvision. The paper does **not** specify the exact NASNet-Edge implementation or DARTS-Compact genotype; attempting to use those names raises a clear `NotImplementedError` rather than silently producing unfaithful comparison architectures. Once verified genotype/source details are available, add their faithful builders to `src/eanas/baselines.py`.

## MLPerf Tiny

MLPerf Tiny is a suite of distinct tasks and official device benchmark rules, **not** a generic CIFAR-10-like dataset. Clone the official repository from https://github.com/mlcommons/tiny and execute a supported official benchmark invocation with:

```bash
python scripts/run_mlperf.py --benchmark-repo /path/to/tiny --command <official benchmark command and arguments>
```

Benchmark applications/model conversion and target-specific firmware must follow the official suite. This repository provides a task-command launcher; it does not purport to reimplement all target-specific embedded benchmark firmware.

## Research reproducibility limits

- Results in the paper (including 95.3% accuracy, 16.8 ms latency and 0.91 J) are **not hardcoded** and may not be reproducible without the original configurations, machines and calibration.
- Arch-A5 is implemented according to the declared block sequence; attention and residual internal implementations are documented, reasonable choices because the manuscript omits sufficient implementation detail.
- Search rewards depend on measured or explicitly estimated energy. Run independent seeds `42 123 2024 3407 7777` to produce real mean/SD/95% CI. Raw per-seed records are required for significance testing.
- Device speed and energy results must be reported separately by platform and backend.
- Full 1000-candidate NAS, ImageNet training and five-seed ablations require considerable hardware and time. Smoke tests do not validate scientific performance.

See `docs/IMPLEMENTATION_NOTES.md` for methodological assumptions.

## Export and statistical summaries

```bash
pip install onnx
python scripts/export_onnx.py --checkpoint results/final/cifar10_arch_a5_seed42/best.pt --out checkpoints/arch_a5.onnx
python scripts/summarize_runs.py --csv my_independent_run_results.csv --reference EANAS
```

ONNX export is optional and requires the `onnx` package. Statistical summaries require actual independent per-seed records.
