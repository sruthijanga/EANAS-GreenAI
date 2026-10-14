[![DOI](https://zenodo.org/badge/1418504874.svg)](https://doi.org/10.5281/zenodo.23360559)

# EANAS-GreenAI
### Energy-Aware Neural Architecture Search for Edge-AI Deployments

**EANAS-GreenAI** is a research implementation of *Green AI: Energy-Aware Neural Architecture Search for Edge-AI Deployments*. The framework explores neural network architectures by balancing predictive accuracy, inference latency, and energy consumption to identify architectures suitable for resource-constrained edge-AI environments.

The implementation combines a five-block convolutional neural network (CNN) search space, reinforcement learning, evolutionary optimization, multi-objective reward evaluation, Pareto-based architecture selection, final model retraining, device profiling, and statistical analysis.

> **Research integrity:** Synthetic data and calibrated energy estimates are intended for software verification or explicitly labelled surrogate experiments. They must not be presented as empirical benchmark results. Actual accuracy, latency, and energy measurements must be obtained through appropriate experiments.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Optimization Objectives](#optimization-objectives)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CIFAR-10 Architecture Search](#cifar-10-architecture-search)
- [Final Model Training](#final-model-training)
- [Baseline Comparisons](#baseline-comparisons)
- [Ablation Studies](#ablation-studies)
- [Device Profiling](#device-profiling)
- [MLPerf Tiny Integration](#mlperf-tiny-integration)
- [ONNX Export](#onnx-export)
- [Statistical Analysis](#statistical-analysis)
- [Experimental Reproducibility](#experimental-reproducibility)
- [Limitations](#limitations)
- [Research Ethics and Reporting](#research-ethics-and-reporting)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **CNN architecture search:** Searches candidate architectures using a five-block CNN design space.
- **Reinforcement learning:** Uses an LSTM-based policy and REINFORCE to explore architecture configurations.
- **Evolutionary optimization:** Supports mutation, crossover, and tournament selection.
- **Multi-objective optimization:** Balances accuracy, inference latency, and energy consumption.
- **Pareto-based selection:** Identifies candidate architectures representing trade-offs among competing objectives.
- **Final model retraining:** Supports retraining selected architectures on CIFAR-10 and ImageNet.
- **Benchmark comparisons:** Provides MobileNetV2 and EfficientNet-B0 baseline implementations.
- **Ablation experiments:** Supports evaluating search variants, including configurations without reinforcement learning.
- **Device profiling:** Measures inference latency, throughput, parameter storage size, and energy when synchronized power measurements are available.
- **Statistical reporting:** Summarizes independent experimental runs using mean, standard deviation, and confidence intervals.
- **ONNX export:** Supports optional export of trained models for compatible inference workflows.
- **MLPerf Tiny integration:** Provides a launcher for supported commands from the official MLPerf Tiny repository.
- **Smoke testing:** Allows basic functional verification using synthetic data without downloading datasets.

---

## System Architecture

The EANAS-GreenAI workflow is organized into the following stages:

1. **Search-space definition:** Construct candidate CNN architectures from the supported five-block design space.
2. **Architecture exploration:** Generate and evaluate candidate configurations.
3. **Reinforcement learning:** Use an LSTM policy with REINFORCE to guide architecture selection.
4. **Evolutionary optimization:** Apply mutation, crossover, and tournament selection to improve candidates.
5. **Multi-objective evaluation:** Evaluate predictive accuracy, latency, and energy-related objectives.
6. **Pareto selection:** Identify non-dominated architectures and select a suitable candidate under the configured constraints.
7. **Final retraining:** Train the selected architecture using the intended dataset and training configuration.
8. **Device profiling:** Measure inference performance and, when supported by synchronized instrumentation, energy consumption.
9. **Statistical evaluation:** Aggregate results from independent random seeds and compare experimental configurations.

### Workflow overview

```text
Dataset / Synthetic Test Data
            |
            v
   Five-Block CNN Search Space
            |
            v
 Reinforcement Learning (LSTM)
            |
            v
 Evolutionary Optimization
 Mutation / Crossover / Tournament
            |
            v
 Multi-Objective Reward Evaluation
 Accuracy / Latency / Energy
            |
            v
     Pareto-Based Selection
            |
            v
    Selected Architecture
            |
            v
       Final Retraining
            |
            v
 Device Profiling and Benchmarking
            |
            v
 Statistical Analysis and Reporting
```

---

## Optimization Objectives

EANAS-GreenAI considers three primary objectives.

| Objective | Description | Optimization direction |
|---|---|---|
| Predictive accuracy | Classification performance on the target dataset | Maximize |
| Inference latency | Time required to process an inference request | Minimize |
| Energy consumption | Energy used during the defined inference measurement window | Minimize |

A normalized reward combines these objectives according to the configured reference values, weights, and constraints.

The exact reward calculation, normalization method, and constraint handling should be documented in the implementation and configuration files. The reference values in `configs/search.yaml` are implementation choices unless independently verified against the manuscript.

### Energy measurement modes

**Measured energy**

Uses an externally collected, synchronized power measurement stream. Energy is calculated by integrating power over the defined measurement interval.

**Calibrated energy**

Uses a configured power estimate to approximate energy consumption. This mode is useful for surrogate experiments and software testing but does not replace direct device measurements.

**Synthetic testing**

Uses synthetic data to validate software functionality without requiring a real dataset download. Results from these tests are not empirical model-performance findings.

---

## Project Structure

The repository is organized around the following implementation components. The exact file inventory should match the files committed to the repository.

```text
EANAS-GreenAI/
├── configs/
│   └── search.yaml
├── docs/
│   └── IMPLEMENTATION_NOTES.md
├── scripts/
│   ├── run_search.py
│   ├── generate_results.py
│   ├── train_final.py
│   ├── profile_device.py
│   ├── run_baselines.py
│   ├── run_ablations.py
│   ├── run_mlperf.py
│   ├── export_onnx.py
│   └── summarize_runs.py
├── src/
│   └── eanas/
│       └── baselines.py
├── tests/
├── results/
├── checkpoints/
├── requirements.txt
├── pyproject.toml
└── README.md
```

The tree above is an illustrative overview of the repository's intended layout. Confirm that every listed file and directory exists before publishing the README.

---

## Requirements

Recommended software environment:

- Python 3.10
- PyTorch 2.2
- torchvision 0.17
- Git
- pytest
- Optional CUDA-compatible GPU and suitable drivers
- Optional ONNX package for model export
- Optional external power measurement equipment for measured-energy experiments

The listed Python package versions are recommended for matching the intended experimental environment. Compatibility depends on the operating system, hardware, CUDA version, and available package builds.

### Hardware considerations

CPU execution is suitable for basic functional testing and smaller experiments. GPU acceleration is recommended for computationally intensive architecture searches and large-scale training.

Actual edge-device measurements require access to the target hardware. Results from a desktop CPU or GPU must not be reported as measurements from an untested Raspberry Pi, NVIDIA Jetson, or another edge device.

---

## Installation

### 1. Clone the repository

```bash
git clone <[YOUR_GITHUB_REPOSITORY_URL](https://github.com/sruthijanga/EANAS-GreenAI)>
cd EANAS-GreenAI
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### 2. Create a virtual environment

**Windows — Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Install a compatible PyTorch and torchvision build for your platform. For GPU execution, select a supported build compatible with the installed NVIDIA driver and the relevant PyTorch distribution.

### 4. Run tests

```bash
pytest -q
```

A passing test suite confirms that the tests executed successfully. It does not independently establish scientific validity or reproduce the manuscript's reported results.

---

## Quick Start

The following commands perform smoke tests and basic functional verification without requiring CIFAR-10 or ImageNet downloads.

### Step 1: Verify the architecture search

```bash
python scripts/run_search.py \
  --smoke \
  --synthetic \
  --energy-mode calibrated \
  --calibrated-dynamic-power-w 2.0
```

### Step 2: Generate smoke-test results

```bash
python scripts/generate_results.py \
  --result-dir results/smoke
```

### Step 3: Verify the training pipeline

```bash
python scripts/train_final.py \
  --synthetic \
  --epochs 1 \
  --max-batches 1 \
  --workers 0
```

### Step 4: Verify device profiling

```bash
python scripts/profile_device.py \
  --warmup 2 \
  --runs 10
```

These commands assume that the repository's command-line arguments match the examples and that the required dependencies are installed.

**Important:** The calibrated power value of `2.0 W` is an example configuration, not a measured device property. Do not use it as evidence of actual energy efficiency unless independently justified and clearly reported as an estimate.

---

## CIFAR-10 Architecture Search

For a real dataset experiment, configure the search settings in `configs/search.yaml` before execution.

Review the following parameters:

- Compute device and backend
- Architecture search budget
- Candidate evaluation limits
- Accuracy, latency, and energy reference values
- Constraint thresholds
- Random seed
- Dataset and data directory
- Energy measurement mode

### Run with measured energy

```bash
python scripts/run_search.py \
  --config configs/search.yaml \
  --energy-mode measured \
  --power-csv measurement.csv
```

The `measurement.csv` file must contain numeric columns named:

```csv
timestamp_s,power_w
0.000,2.10
0.010,2.15
0.020,2.12
```

The rows above are illustrative only and are not actual measurement results.

The power data must:

1. Come from the external measurement equipment.
2. Use timestamps synchronized to the process's monotonic clock or a correctly aligned equivalent time reference.
3. Cover the complete inference measurement window.
4. Use documented sampling intervals, device power settings, and measurement procedures.
5. Be processed using a documented energy-integration method.

If the measurement stream is missing, unsynchronized, or incomplete, energy claims are not scientifically supported.

### Run with calibrated energy

For an explicitly labelled surrogate experiment:

```bash
python scripts/run_search.py \
  --config configs/search.yaml \
  --energy-mode calibrated \
  --calibrated-dynamic-power-w 2.0
```

Calibrated energy is an approximation based on the selected power assumption. Report it separately from directly measured energy.

---

## Final Model Training

### CIFAR-10

```bash
python scripts/train_final.py \
  --architecture arch_a5 \
  --dataset cifar10 \
  --data-root data \
  --epochs 200 \
  --device cuda
```

The configured CIFAR-10 workflow uses 40,000 training samples and 10,000 validation samples from the official training partition, while keeping the official test set held out for final evaluation.

Ensure that the implementation follows this split and does not use the test set for model selection or hyperparameter tuning.

### ImageNet

```bash
python scripts/train_final.py \
  --architecture arch_a5 \
  --dataset imagenet \
  --data-root /path/to/imagenet \
  --epochs 100 \
  --device cuda
```

The expected directory structure is:

```text
imagenet/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

Replace the example class names with the actual ImageNet class directories.

ImageNet access, licensing, dataset preparation, storage, and redistribution are the user's responsibility. This repository does not automatically provide ImageNet data.

The `arch_a5` identifier must correspond to an architecture registered in the implementation. Confirm that the selected architecture matches the intended design before training.

---

## Baseline Comparisons

The baseline runner supports the following torchvision architectures:

- MobileNetV2
- EfficientNet-B0

Run:

```bash
python scripts/run_baselines.py \
  --models mobilenetv2 efficientnet_b0
```

For a scientifically valid comparison, use consistent dataset partitions, evaluation metrics, input preprocessing, device conditions, and measurement procedures wherever applicable.

### NASNet-Edge and DARTS-Compact

The manuscript does not specify sufficient source or genotype details to establish faithful implementations of NASNet-Edge and DARTS-Compact.

The current implementation intentionally raises `NotImplementedError` for these unsupported architectures rather than silently substituting unrelated models.

To add these baselines, provide verified architecture definitions, source references, configuration details, and compatible evaluation procedures in `src/eanas/baselines.py`.

---

## Ablation Studies

Ablation experiments help evaluate the contribution of individual components in the search framework.

Example command:

```bash
python scripts/run_ablations.py \
  --variant without_rl \
  --synthetic \
  --smoke \
  --energy-mode calibrated \
  --calibrated-dynamic-power-w 2.0
```

This example validates an ablation workflow using synthetic data. It does not establish the performance contribution of reinforcement learning.

For research experiments, use real data and consistent experimental settings. Compare relevant variants, such as:

- Full framework
- Search without reinforcement learning
- Search without evolutionary optimization
- Alternative reward configurations
- Alternative Pareto-selection settings

Only include variants that are actually implemented and verified in the repository.

---

## Device Profiling

The profiler measures processing latency, inference throughput, and parameter storage size. Energy measurement is supported when a suitable synchronized power stream is available.

Example:

```bash
python scripts/profile_device.py \
  --checkpoint results/final/cifar10_arch_a5_seed42/best.pt \
  --device cpu \
  --warmup 50 \
  --runs 1000
```

### Measured-energy profiling

Where supported by the implementation:

```bash
python scripts/profile_device.py \
  --checkpoint results/final/cifar10_arch_a5_seed42/best.pt \
  --device cpu \
  --warmup 50 \
  --runs 1000 \
  --power-csv measurement.csv \
  --idle-w <MEASURED_IDLE_POWER>
```

Replace `<MEASURED_IDLE_POWER>` with an actual idle-power measurement in watts. Confirm the exact argument syntax with:

```bash
python scripts/profile_device.py --help
```

Record the device model, processor, operating system, software versions, inference backend, power mode, batch size, warm-up procedure, run count, sampling method, and measurement boundaries.

Without an appropriate power stream, energy should remain unavailable rather than being presented as a measured result. If calibrated estimates are used, label them explicitly.

---

## MLPerf Tiny Integration

MLPerf Tiny is a benchmark suite for tiny machine-learning systems and embedded deployments. It contains task-specific workloads and official benchmark requirements; it is not a generic CIFAR-10 replacement.

Official repository:

https://github.com/mlcommons/tiny

Example launcher:

```bash
python scripts/run_mlperf.py \
  --benchmark-repo /path/to/tiny \
  --command "<official benchmark command and arguments>"
```

Replace the path and command with a supported invocation from the official repository. Verify the launcher's quoting and argument-forwarding behavior using its help output.

Benchmark applications, model conversion, target-specific firmware, and device evaluation must follow the relevant official benchmark instructions.

The EANAS-GreenAI launcher does not independently reimplement all MLPerf Tiny workloads, firmware, or compliance procedures.

---

## ONNX Export

ONNX export is optional and requires the `onnx` package.

Install it:

```bash
pip install onnx
```

Export a trained checkpoint:

```bash
python scripts/export_onnx.py \
  --checkpoint results/final/cifar10_arch_a5_seed42/best.pt \
  --out checkpoints/arch_a5.onnx
```

The checkpoint must exist and contain a model compatible with the export script.

Verify the exported graph and evaluate the model using the intended runtime before treating the ONNX file as a deployment artifact.

---

## Statistical Analysis

Run summaries on actual independent experimental records:

```bash
python scripts/summarize_runs.py \
  --csv my_independent_run_results.csv \
  --reference EANAS
```

The input CSV must contain the fields expected by the summarization script, including the relevant metric values and independent-run identifiers.

### Recommended independent seeds

```text
42
123
2024
3407
7777
```

For each seed, preserve the raw result records and record the configuration, software environment, hardware platform, and energy measurement method.

A confidence interval should be calculated from the independent experimental units appropriate to the research design. Repeated inference measurements from a single trained model should not automatically be treated as independent training runs.

Statistical significance tests require suitable raw observations and a justified testing procedure. Summary statistics alone are not sufficient for every comparison.

---

## Experimental Reproducibility

For reproducible experiments:

1. Pin the software dependencies and record the Python, PyTorch, torchvision, CUDA, and driver versions.
2. Save the full search configuration and training configuration.
3. Record the random seed for each independent run.
4. Preserve raw per-seed results and trained checkpoints as appropriate.
5. Keep the test set separate from architecture selection and hyperparameter tuning.
6. Document the device, backend, batch size, latency methodology, and power measurement setup.
7. Separate directly measured energy from calibrated estimates.
8. Report failed runs, missing measurements, and deviations from the intended protocol.
9. Distinguish smoke-test validation from empirical performance evaluation.
10. Document implementation choices that are not fully specified by the manuscript.

These values are manuscript-reported reference figures, not guaranteed outputs of this implementation. They must not be described as independently reproduced results without corresponding experimental evidence.

The implementation includes reasonable choices for architecture internals where the manuscript does not supply sufficient detail. In particular, attention and residual-block details should be checked against `docs/IMPLEMENTATION_NOTES.md`.

Exact reproduction may require original configurations, architecture definitions, hardware, software versions, measurement procedures, and trained checkpoints that are not available in the manuscript.

---

## Limitations

- Synthetic smoke tests do not establish empirical predictive performance.
- Calibrated energy is an estimate, not direct energy measurement.
- Search rewards depend on reference values, normalization, and measurement quality.
- Actual latency and energy vary by hardware, backend, device state, and power settings.
- Large architecture searches, ImageNet training, and multi-seed ablations can require substantial computational resources.
- NASNet-Edge and DARTS-Compact are not claimed as faithful implementations without verified source definitions.
- ONNX export does not by itself guarantee deployment compatibility or performance.
- The presence of an MLPerf Tiny launcher does not imply official benchmark compliance.
- The manuscript's reported accuracy, latency, and energy figures are not guaranteed to be reproduced.

---

## Research Ethics and Reporting

EANAS-GreenAI is intended for research, experimentation, and engineering evaluation.

When publishing or presenting results:

- Identify synthetic, estimated, and measured values separately.
- Do not invent hardware measurements or benchmark outcomes.
- Report the actual experimental conditions.
- Include relevant configurations and independent-run records.
- Disclose methodological assumptions and implementation differences.
- Avoid claiming exact manuscript reproduction without supporting evidence.

---

## Contributing

Contributions that improve correctness, reproducibility, documentation, testing, and deployment support are welcome.

Suggested workflow:

1. Fork the repository.
2. Create a feature branch.
3. Implement the change and add appropriate tests.
4. Run the test suite.
5. Document any new assumptions, dependencies, or measurement requirements.
6. Submit a pull request describing the changes.

Please ensure that new baseline architectures and benchmark integrations are clearly identified as verified, approximate, or unsupported.

---

## License

Add a `LICENSE` file before publishing the repository publicly. Select a license appropriate to the code's ownership, third-party dependencies, and any manuscript or institutional requirements.

Until a license is added, do not assume that the repository is available for unrestricted reuse, redistribution, or commercial use.

---

## Acknowledgments and References

- **Research manuscript:** *Green AI: Energy-Aware Neural Architecture Search for Edge-AI Deployments*.
- **PyTorch:** https://pytorch.org/
- **torchvision:** https://pytorch.org/vision/
- **ONNX:** https://onnx.ai/
- **MLPerf Tiny:** https://github.com/mlcommons/tiny

Add the manuscript's formal citation, DOI or publication URL, and relevant baseline architecture references when available.

---

**EANAS-GreenAI — Toward accuracy-aware, latency-aware, and energy-conscious neural architecture search for edge AI.**
