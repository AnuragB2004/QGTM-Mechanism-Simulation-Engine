# QGTM – Quantum Game-Theoretic Mechanism for Multi-User Quantum Networks
### Hardware Validation Suite on IBM Quantum Devices

> **Paper:** *Incentive-Compatible Resource Allocation in Multi-User Quantum Networks via Quantum Game Theory*  
> **Authors:** Anurag Bhattacharjee, Anjan Bandyopadhyay — KIIT University  
> **Venue:** IEEE Journal (under submission)

---

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Running Experiments](#running-experiments)
   - [Local Simulation (no IBM account needed)](#local-simulation)
   - [IBM Quantum Hardware](#ibm-quantum-hardware)
   - [Multi-Backend Cross-Comparison](#multi-backend-cross-comparison)
6. [Experiment Details](#experiment-details)
7. [Results Reproduction](#results-reproduction)
8. [LaTeX Integration](#latex-integration)
9. [Testing](#testing)
10. [Supported Backends](#supported-backends)
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

This repository contains the complete implementation of the **Quantum
Game-Theoretic Mechanism (QGTM)** for incentive-compatible resource allocation
in multi-user quantum networks, along with a hardware validation suite that runs
on real IBM Quantum 5-qubit devices (`ibmq_lima`, `ibmq_belem`, `ibmq_manila`,
etc.).

The mechanism exploits two uniquely quantum phenomena:
- **Measurement disturbance** – lying about one's quantum state collapses shared
  entanglement, reducing the liar's own payoff.
- **Quantum commitment** – entangled states are physically impossible to forge.

Together these enforce truthful demand reporting as a **dominant strategy**,
improving social welfare by ~18%, Jain's fairness index by ~22%, and reducing
the Price of Anarchy from 2.41 to 1.09 (simulation) / ≤1.17 (hardware).

---

## Repository Structure

```
qgtm-hardware/
├── src/
│   ├── qgtm_hardware_experiments.py   # Main experiment suite (all 6 experiments)
│   ├── multi_backend_runner.py        # Cross-device comparison runner
│   └── tests/
│       └── test_qgtm.py              # Pytest unit + integration tests
├── latex/
│   └── sec_hardware_results.tex      # Drop-in LaTeX section for the paper
├── results/                          # JSON result files (auto-created)
├── figures/                          # PDF/PNG plots (auto-created)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AnuragB2004/QGTM-Mechanism-Simulation-Engine.git
cd QGTM-Mechanism-Simulation-Engine

# 2. Install
pip install -r requirements.txt

# 3. Run all experiments locally (no IBM token needed)
python src/qgtm_hardware_experiments.py --simulate --quick

# 4. Run on IBM hardware
python src/qgtm_hardware_experiments.py \
    --backend ibmq_manila \
    --token YOUR_IBM_TOKEN \
    --shots 1024
```

Results (JSON) are saved in `results/` and figures (PDF) in `figures/`.

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | ≥ 3.10  | 3.11 recommended |
| pip         | ≥ 23    | `pip install --upgrade pip` |
| IBM Quantum account | Optional | Required for hardware runs only |

### Steps

```bash
# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "
import qiskit, qiskit_aer
print('Qiskit:', qiskit.__version__)
print('Aer:   ', qiskit_aer.__version__)
"
```

---

## Running Experiments

### Local Simulation

No IBM account is needed. The Aer simulator can run in two modes:

```bash
# Noiseless simulation
python src/qgtm_hardware_experiments.py --simulate

# Device-realistic noise (models ibmq_lima profile)
python src/qgtm_hardware_experiments.py --simulate --noise

# Quick test (fewer trials, faster)
python src/qgtm_hardware_experiments.py --simulate --quick

# Custom output directories
python src/qgtm_hardware_experiments.py \
    --simulate --noise \
    --results-dir my_results \
    --figures-dir my_figures
```

### IBM Quantum Hardware

#### Get your API token
1. Sign up at [quantum.ibm.com](https://quantum.ibm.com)
2. Go to **My Account → API token → Copy**

#### Run on a specific backend

```bash
# ibmq_manila (lowest noise, recommended)
python src/qgtm_hardware_experiments.py \
    --backend ibmq_manila \
    --token YOUR_TOKEN \
    --shots 1024

# ibmq_lima
python src/qgtm_hardware_experiments.py \
    --backend ibmq_lima \
    --token YOUR_TOKEN

# ibmq_belem
python src/qgtm_hardware_experiments.py \
    --backend ibmq_belem \
    --token YOUR_TOKEN
```

#### Using environment variable (recommended for CI/CD)

```bash
export IBM_QUANTUM_TOKEN="your_token_here"
python src/qgtm_hardware_experiments.py --backend ibmq_manila
```

### Multi-Backend Cross-Comparison

Runs noise characterisation + PoA experiment on all 4 supported backends
and produces a cross-device comparison figure and LaTeX table:

```bash
# All devices (hardware)
python src/multi_backend_runner.py --token YOUR_TOKEN

# All devices (simulated with device-realistic noise)
python src/multi_backend_runner.py --simulate

# Custom shots
python src/multi_backend_runner.py --simulate --shots 2048
```

---

## Experiment Details

The suite runs 6 experiments corresponding to Figures 7–11 in the paper:

| ID | Experiment | Key Variables | Circuits/run |
|----|-----------|---------------|-------------|
| E1 | Social welfare vs N | N ∈ {2,3,4,5}, q_s=0.3 | Allocation + SWAP-test |
| E2 | Fairness vs selfish fraction | q_s ∈ [0, 0.6], N=4 | Allocation + SWAP-test |
| E3 | Price of Anarchy comparison | N=4, q_s=0.3, κ=2 | Allocation + SWAP-test |
| E4 | Penalty factor κ sensitivity | κ ∈ {0.5,1,2,3,5}, N=4 | Allocation + SWAP-test |
| E5 | Fidelity vs depolarising noise | p ∈ [0.01, 0.10], N=3 | Noisy allocation |
| E6 | Hardware noise characterisation | All backends | Bell benchmark |

### Circuit descriptions

#### Allocation Circuit (N users)
```
|0⟩_arb ─── J(γ) ─── U₁(θ₁,φ₁) ─── J†(γ) ─── M
|0⟩_U1  ─── J(γ) ─── U₁'         ─── J†(γ) ─── M
...
|0⟩_UN  ─── J(γ) ─── Uₙ(θₙ,φₙ) ─── J†(γ) ─── M
```
- **J(γ)**: entangling operator (Hadamard + cascaded CNOTs + RZ)
- **Uᵢ(θ,φ)**: user strategy unitary (RZ · RY · RZ decomposition)
- **Measurement** → allocation probabilities

#### SWAP Test (Quantum Verification)
```
|0⟩_anc ─── H ─── ●───── H ─── M
|ψ_actual⟩ ──── × ──
|ψ_claimed⟩──── × ──
```
P(ancilla=0) = (1 + F(ρ_actual, ρ_claimed)) / 2  
→ δᵢ = ½‖ρ_actual − ρ_claimed‖₁ ≈ √(1 − F)

### Strategy encoding

| User type | Demand ratio η | θ | φ |
|-----------|---------------|---|---|
| Truthful | η = 1.0 | 0 | 0 |
| Mild misreport | η ∈ (1.0, 1.5) | 0..π/2 | 0 |
| Strong misreport | η ∈ [1.5, 3.0] | π/2..π | π/2 |

---

## Results Reproduction

To reproduce the exact numbers from the paper (Table III and Figures 7–11):

```bash
# Full run, 10 trials per experiment, 1024 shots
python src/qgtm_hardware_experiments.py \
    --backend ibmq_manila \
    --token YOUR_TOKEN \
    --shots 1024

# Then cross-backend comparison for Table in hardware section
python src/multi_backend_runner.py \
    --token YOUR_TOKEN \
    --shots 1024 \
    --results-dir results \
    --figures-dir figures
```

Expected results (ibmq_manila, N=4, q_s=0.3, κ=2.0):

| Metric | Simulation | Hardware (manila) | Degradation |
|--------|-----------|-------------------|-------------|
| Social welfare W | 0.932 | 0.901 ± 0.012 | −3.3% |
| Fairness J | 0.931 | 0.914 ± 0.009 | −1.8% |
| Avg. fidelity F̄ | 0.903 | 0.878 ± 0.008 | −2.8% |
| PoA | 1.09 | 1.11 ± 0.04 | +1.8% |
| SGR (%) | 84.2 | 81.4 ± 1.8 | −3.3% |

> **Note:** Hardware results include shot noise (1024 shots) and device noise
> (depolarising p ≈ 0.066 on manila). Run with `--shots 4096` for tighter
> confidence intervals.

---

## LaTeX Integration

### Drop-in hardware section

The file `latex/sec_hardware_results.tex` is a self-contained LaTeX section
(Section X of the paper) that can be inserted directly:

```latex
% In your main .tex file, after \section{Results and Analysis}:
\input{latex/sec_hardware_results}
```

It uses the same packages already declared in the paper preamble
(`tikz`, `pgfplots`, `booktabs`, `tcolorbox`, `multirow`).

### Auto-generated tables

Each experiment run produces a LaTeX table file in `results/`:

```bash
# After running experiments:
ls results/*.tex
# → results/20241201_143022_hw_results_table.tex
# → results/cross_backend_table.tex
```

Include in your paper:
```latex
\input{results/hw_results_table}
\input{results/cross_backend_table}
```

### Figures

PDF figures are saved in `figures/` and can be included with:
```latex
\includegraphics[width=\columnwidth]{figures/fig_poa_hardware.pdf}
\includegraphics[width=\columnwidth]{figures/fig_fidelity_noise_hardware.pdf}
\includegraphics[width=\columnwidth]{figures/fig_cross_backend.pdf}
```

---

## Testing

```bash
# Run all tests
pytest src/tests/test_qgtm.py -v

# Run with coverage
pytest src/tests/test_qgtm.py -v --cov=src --cov-report=html

# Run a specific test class
pytest src/tests/test_qgtm.py::TestQGTMCircuitBuilder -v
pytest src/tests/test_qgtm.py::TestIntegration -v

# Quick smoke test (fast subset)
pytest src/tests/test_qgtm.py -v -k "not integration"
```

### Test categories

| Class | Tests | Description |
|-------|-------|-------------|
| `TestQGTMCircuitBuilder` | 8 | Circuit construction, unitarity, Bell state |
| `TestQGTMMechanism` | 10 | Allocation, SWAP-test, penalty, DSIC property |
| `TestQGTMBackend` | 6 | Simulator init, circuit execution, noise |
| `TestIntegration` | 4 | End-to-end experiment E1, E3, E5 + DSIC |

---

## Supported Backends

| Backend | Qubits | Status | Notes |
|---------|--------|--------|-------|
| `simulator` | Unlimited | ✅ Always available | Noiseless Aer |
| `ibmq_lima` | 5 | ✅ Active | Falcon r4, good reliability |
| `ibmq_belem` | 5 | ✅ Active | Falcon r4, slightly noisier |
| `ibmq_manila` | 5 | ✅ Active | Falcon r4T, lowest noise |
| `ibmq_quito` | 5 | ✅ Active | Falcon r4 |
| `ibm_nairobi` | 7 | ✅ Active | Eagle r1, 7-qubit (up to N=6) |

> IBM Quantum device availability changes. Check
> [quantum.ibm.com](https://quantum.ibm.com/services/resources) for
> current status. The code automatically lists available backends if the
> requested one is offline.

### Checking available backends

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
for b in service.backends():
    print(b.name, b.num_qubits, b.status().status_msg)
```

---

## Troubleshooting

### `ImportError: No module named 'qiskit_ibm_runtime'`
```bash
pip install qiskit-ibm-runtime
```

### `IBMRuntimeError: 'ibmq_lima' is not a valid backend`
The device may be offline or renamed. List available backends:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
svc = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
print([b.name for b in svc.backends(operational=True)])
```

### `AuthenticationError: Invalid token`
- Verify your token at [quantum.ibm.com](https://quantum.ibm.com)
- Token must not have leading/trailing whitespace
- Use `export IBM_QUANTUM_TOKEN="..."` (with quotes for tokens containing `+`)

### `CircuitError: Cannot map circuit to backend`
The requested circuit exceeds the backend's qubit count. Use `--backend simulator`
for N > 4, or switch to a larger device (e.g., `ibm_nairobi` for N ≤ 6).

### Slow queue times
IBM Quantum free-tier jobs can queue for hours. Options:
- Use `--simulate --noise` for immediate results with realistic noise
- Use IBM Quantum Premium instances for priority access
- Use the Aer `ibmq_lima` noise model, which is calibrated from real hardware

### Matplotlib backend error (`TclError`)
On headless servers, set the backend before importing:
```bash
export MPLBACKEND=Agg
python src/qgtm_hardware_experiments.py --simulate
```
The code already sets `matplotlib.use("Agg")` at import time.

---

## Project Dependencies

```
qiskit >= 1.0          – Quantum circuit construction and transpilation
qiskit-aer >= 0.14     – Local simulation with realistic noise models
qiskit-ibm-runtime >= 0.20 – IBM Quantum cloud access
numpy >= 1.24          – Numerical arrays
scipy >= 1.10          – Statistics (sem, t-distribution)
matplotlib >= 3.7      – Publication-quality figures
pytest >= 7.4          – Unit and integration tests
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bhattacharjee2024qgtm,
  title   = {Incentive-Compatible Resource Allocation in Multi-User Quantum
             Networks via Quantum Game Theory},
  author  = {Bhattacharjee, Anurag and Bandyopadhyay, Anjan},
  journal = {IEEE Transactions on Quantum Engineering},
  year    = {2024},
  note    = {Under review}
}
```

---

## License

This project is released under the **MIT License**.  
See `LICENSE` for details.

---

## Acknowledgements

The authors thank the IBM Quantum team for providing access to
\texttt{ibmq\_lima}, \texttt{ibmq\_belem}, and \texttt{ibmq\_manila}
through the IBM Quantum Network. The hardware experiments were conducted
using the \texttt{qiskit-ibm-runtime} SDK.

---

*Last updated: 2024 · Anurag Bhattacharjee · KIIT University*
