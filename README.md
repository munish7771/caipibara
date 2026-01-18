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
