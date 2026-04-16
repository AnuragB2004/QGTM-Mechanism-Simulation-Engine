# QGTM Mechanism Simulation Engine

Quantum Game-Theoretic Mechanism (QGTM) implementation for incentive-compatible
resource allocation in multi-user quantum networks.

This repository contains the hardware validation suite and experiment code used
to evaluate the QGTM mechanism on real IBM Quantum devices and local Qiskit
simulations.

## Repository Structure

- `qgtm_hardware/`
  - `src/`
    - `qgtm_hardware_experiments.py` — main experiment suite
    - `multi_backend_runner.py` — cross-backend comparison runner
    - `tests/test_qgtm.py` — Pytest tests
  - `requirements.txt` — Python dependencies for the hardware experiment suite
  - `results/` — generated JSON result files
  - `figures/` — generated plot output
  - `latex/` — LaTeX sections for paper integration
  - `README.md` — package-specific documentation

## Overview

The QGTM mechanism uses quantum properties to ensure truthful demand reporting
in multi-user quantum networks. The hardware validation suite is designed to
measure how the mechanism performs in practice on IBM Quantum backends and in
simulator environments.

Key capabilities:
- local simulation with or without noise
- hardware validation on IBM Quantum devices such as `ibmq_manila`,
  `ibmq_lima`, and `ibmq_belem`
- cross-backend comparison for fairness, welfare, price of anarchy, and noise
  sensitivity

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r qgtm_hardware/requirements.txt
```

## Running locally

```bash
python qgtm_hardware/src/qgtm_hardware_experiments.py --simulate
```

For a quick execution mode:

```bash
python qgtm_hardware/src/qgtm_hardware_experiments.py --simulate --quick
```

## Running on IBM Quantum hardware

Set your IBM Quantum token and specify a backend:

```bash
set IBM_QUANTUM_TOKEN=YOUR_TOKEN
python qgtm_hardware/src/qgtm_hardware_experiments.py --backend ibmq_manila --token %IBM_QUANTUM_TOKEN%
```

## Cross-backend experiments

```bash
python qgtm_hardware/src/multi_backend_runner.py --token %IBM_QUANTUM_TOKEN%
```

## Results

Experiment outputs are saved into `qgtm_hardware/results/` and plots are saved
into `qgtm_hardware/figures/`.

## Testing

```bash
pytest qgtm_hardware/tests
```

## Notes

- The root `README.md` provides the top-level overview.
- The detailed hardware experiment documentation remains available in
  `qgtm_hardware/README.md`.
