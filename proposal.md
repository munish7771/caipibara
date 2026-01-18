**Repository:** [https://github.com/munish7771/caipibara](https://github.com/munish7771/caipibara)

## 1. Motivation

Explanatory Interactive ML helps fix models that are “right for the wrong reasons,” but too much human feedback might actually hurt learning. If a model keeps training on synthetic counterexamples based on one person’s intuition, it may overfit to that feedback instead of the real data. This project explores when helpful guidance turns into harmful bias.

## 2. Related Topics

This project builds on Explanatory Interactive Learning and the CAIPI framework (Teso & Kersting, 2019), which uses local explanations and counterfactuals as feedback. It relates to lecture topics on LIME and its pitfalls, counterfactual explanations, and work on “Right for the Wrong Reasons” and spurious correlations (Ross et al., 2017).

The CAIPI implementation can be found on GitHub: [https://github.com/stefanoteso/caipi](https://github.com/stefanoteso/caipi).

## 3. Idea

The idea is to test how robust explanatory interactive learning is when human feedback becomes excessive or noisy. By varying the amount of explanation-based counterexamples and simulating imperfect users, we study when this feedback helps the model and when it causes overfitting.

Overfitting to a specific user’s explanations can hurt generalization, causing the model to fail on similar but slightly different groups or scenarios. The goal is to identify when explanation-guided learning stops being helpful and starts introducing bias or brittleness.

### CAIPI Stress-Test Steps

1.  **Initialization**: Start with a dataset split into labeled ($L$) and unlabeled ($U$) sets, a model $f$, a feedback intensity $c$, and a noise probability $p$.
2.  **Loop**: While the labeling budget is not exhausted:
    a.  **Retrain** the model $f$ on the current labeled set $L$.
    b.  **Select Query**: Select the next query $x$ from the unlabeled set $U$ using active learning.
    c.  **Predict & Explain**: Predict $\hat{y} = f(x)$ and generate an explanation $\hat{z}$ for $x$.
    d.  **Get Feedback**: Obtain human feedback: the true label $y$ and the correct explanation $C_{true}$.
    e.  **Simulate Noise**: Inject noise into the explanation with probability $p$ to get $C_{noisy}$.
    f.  **Generate Counterexamples**: Generate $c$ synthetic counterexamples based on $C_{noisy}$.
    g.  **Update Data**: Augment the labeled set $L$ with $(x, y)$ and the synthetic counterexamples, and remove $x$ from $U$.
3.  **Output**: Return the final trained model $f$.

This updated CAIPI does the original algorithm plus two key modifications:
1.  Simulates noisy user feedback with probability $p$.
2.  Allows a tunable number of synthetic counterexamples per feedback, controlled by the feedback intensity $c$.

These changes let us explore the boundary between helpful guidance and overfitting, while the rest of CAIPI—active query selection, retraining, and counterexample generation—remains unchanged.

## 4. Experiments

**Dataset & Metrics:**
*   **Datasets:** Decoy Fashion MNIST and a simple Colors logic dataset with known spurious correlations.
*   **Metrics:** Clean test accuracy, explanation alignment (F1 score), and the generalization gap between training and test performance.

**Experimental Scope:**
*   Run CAIPI with different feedback intensities (e.g., 1 to 100 counterexamples per iteration).
*   Repeat experiments with 5 random seeds.
*   Simulate user errors (10–30% noise).
