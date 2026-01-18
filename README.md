# Caipibara: CAIPI Stress-Test

This project measures the robustness of the CAIPI framework under conditions of excessive or noisy human feedback.

## Overview

See [Project Proposal](proposal.md) for detailed motivation and methodology.

## Structure

*   `src/`: implementation of the modified CAIPI algorithm.
*   `experiments/`: Scripts to run stress tests (varying `c` and `p`).
*   `data/`: Datasets (Decoy FMNIST, Colors).
*   `results/`: Experiment logs and plots.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  (Optional) Install local package:
    ```bash
    pip install -e .
    ```

## Reproduction (RQ2: Passive Evaluation)
To reproduce the baseline results (Fashion MNIST with corner confounders):

**Environment:**
- Python 3.13.5
- Install dependencies: `pip install -r requirements.txt`

**Run Experiment:**
```bash
python src/rq2_repro.py
```
This script will:
1.  Download/Load Fashion MNIST.
2.  Train a baseline MLP (approx 44-48% accuracy).
3.  Train MLPs with Counterexample (CE) augmentation.
4.  Generate `rq2_baseline_results.png`.

