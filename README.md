# Caipibara: CAIPI Stress-Test for Explanatory Interactive Learning

This project investigates the robustness of the **CAIPI** (Counterfactual Active Loop) framework when subjected to **noisy human feedback** and varying **feedback intensities**.
See [Project Proposal](proposal.md) for detailed motivation and methodology.    

## Project Overview

We stress-test the Explanatory Interactive Learning (XIL) loop by:
1.  **Simulating User Noise ($p$)**: We introduce probability $p \in \{0.1, 0.5, 1.0\}$ that the user provides incorrect feedback on the explanation.
2.  **Varying Feedback Intensity ($c$)**: We generate $c \in \{1, 5, 20, 100\}$ synthetic counterexamples for each user correction to test if "more data" helps or hurts under noise.

## Structure

*   `src/`: Source code for the CAIPI algorithm and experiment drivers.
    *   `caipi.py`: Core active learning loop and XIL logic. Forked from [caipi](https://github.com/stefanoteso/caipi)
    *   `run_stress_test.py`: Main script to execute the grid search experiments.
    *   `plot_results.py`: Generates analysis plots (Accuracy, Gap, Recall).
    *   `caipi/`: Custom implementation of the CAIPI algorithm. Forked from [caipi](https://github.com/stefanoteso/caipi)
        *   `image.py`: Custom implementation of the image problem.
*   `proposal.md`: Project proposal.
*   `data/`: Datasets (Decoy Fashion-MNIST).
*   `results/`: Stores experiment logs (`.pickle`) and generated plots (`.png`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/munish7771/caipibara.git
    cd caipibara
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv venv
    # Linux/Mac:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: If automated installation fails for some packages, ensure you have `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `scikit-image`, `lime`, `gensim`, `blessed`, and `requests` installed.*

## Usage

### 1. Running the Stress Test
To run the full grid search of experiments (Noise $\times$ Intensity):

```bash
# make sure you are in the root directory caipibara
# Fast run (for verification, few iterations)
python src/run_stress_test.py --mode fasts
# Prints the experiment plan
python src/run_stress_test.py --mode quarter --explain
# Quarter run (may take 3-4 hours)
python src/run_stress_test.py --mode quarter 
# the results are already stored in results/ so the new run will be skipped. If you want to run it again, delete the results/ directory.
```
*Arguments:*
*   `--mode`: `full`, `quarter` or `fast`.
*   `--explain`: prints the experiment plan. 

### 2. Plotting Results
After the experiments complete, generate the analysis plots:

```bash
python src/plot_results.py
```
This will process the logs in `results/` and save:
*   `results_accuracy_horizontal.png`
*   `results_recall_horizontal.png`
*   `results_gap_horizontal.png`

## Results
The metrics tracked include:
*   **Test Accuracy**: Performance on the clean test set.
*   **Confounder Recall**: Ability of the model to identify the "decoy" artifact correctly.
*   **Generalization Gap**: Difference between training and test accuracy.

## Implementation Details: `src/caipi/image.py`
Significant modifications were made to `src/caipi/image.py` to implement the stress-testing mechanics:

### A. Updated Method Signature
*   **Old**: `def query_corrections(self, i, pred_y, pred_mask, X_test):`
*   **New**: `def query_corrections(..., noise_prob=0.0, feedback_intensity=-1):`
*   **Impact**: Allows passing simulation parameters from the main loop into the problem logic.

### B. Noise Injection Logic (User Errors)
*   **Mechanism**: With probability `p=noise_prob`, the system "hallucinates" feedback. It randomly selects pixels that are part of the *object* (valid prediction) but NOT part of the confounder, and marks them as confounders.
*   **Effect**: Simulates a user incorrectly teaching the model that a valid part of the object (e.g., a bag handle) is a confounder. Tests resilience to bad teaching signals.

### C. Feedback Intensity Logic (Variable Strength)
*   **Old**: Hardcoded loop `for value in [-10, 0, 11]`, generating exactly 3 counterexamples.
*   **New**: Dynamic list generation. If `feedback_intensity > 0`, it expands the list of replacement values (starting with `[250, 0, 11]` and adding random `uint8` values) until it matches the requested `intensity`.
*   **Impact**: Enables generating anywhere from 1 to 100+ counterexamples per interaction to test "data flooding".

### D. FashionProblem Overhaul
*   **Decoy Dataset**: The `FashionProblem` class was rewritten to implement "Decoy Fashion MNIST", filtering for 5 classes and placing specific pixel patches as confounders based on class label.
*   **Mask Handling**: `ImageProblem` now accepts pre-computed `confounder_masks` in `__init__`, required for the specific patch-based confounders.
*   **Robustness**: Added `try/except` blocks around LIME explainers to prevent experiment crashes on singular matrix errors.

## Implementation Details: `src/caipi.py` (Core Logic)
The core active learning driver was modified (forked from `caipi.py`) to support the stress test:

### A. Argument Parsing
*   Added `--noise-prob`: Float argument to specify the probability of incorrect feedback.
*   Added `--feedback-intensity`: Integer argument to control the number of counterexamples.
*   Added `--snapshot-dir`: Support for saving/resuming experiments to handle long runtimes.

### B. Interactive Loop Integration
*   **Signature Update**: The `caipi()` function now accepts `noise_prob` and `feedback_intensity`.
*   **Feedback Passing**: Inside the active learning loop, these parameters are passed to `problem.query_corrections(...)`.
*   **Filename Logic**: `_get_basename` was updated to include `np=` and `fi=` tags, ensuring unique log files for every configuration in the grid search.
