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

### Setup on macOS
Run the following commands to set up the environment from scratch:

```bash
# 1. Clone the repository
git clone https://github.com/munish7771/caipibara.git
cd caipibara

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install numpy scipy scikit-learn matplotlib scikit-image lime gensim blessed requests

# 4. Download datasets
python3 src/download_fmnist.py

# 5. Verify installation with a fast run
python3 src/run_stress_test.py --mode fast
```

## Reproduction (RQ2: Passive Evaluation)
To reproduce the baseline results (Fashion MNIST with corner confounders):

**Environment:**
- Python 3.11.14
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

