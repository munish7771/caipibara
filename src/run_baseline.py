#!/usr/bin/env python3
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

# Import the necessary components from the main script
# This ensures we use the EXACT same model, data loading, and evaluation logic
from src.run_caipi import PROBLEMS, eval_passive

class BaselineArgs:
    """Mock arguments object to pass to eval_passive"""
    def __init__(self):
        # Core Settings
        self.problem = 'fashion'
        self.learner = 'mlp'
        self.strategy = 'random' # Not used in passive
        self.seed = 0
        
        # Data Config (Matching Stress Test 'quarter' mode)
        self.n_examples = 2000
        self.n_folds = 10
        self.prop_eval = 0.05
        
        # THE REQUESTED CONFIGURATION
        self.noise_prob = 0.0          # Zero Noise
        self.feedback_intensity = 1    # 1 Counterexample
        
        # Problem Params (Defaults)
        self.corr_type = None
        self.n_samples = 200    # Quarter mode: 200
        self.n_features = 3     # Quarter mode: 3
        self.kernel_width = 0.75
        self.lime_repeats = 1
        self.vectorizer = None

def run_caipi_baseline():
    print("--- Running Standardized CAIPI Baseline ---")
    print("Configuration: Noise=0.0, Intensity=1")
    
    args = BaselineArgs()
    rng = np.random.RandomState(args.seed)
    
    print(f"Initializing {args.problem} problem with n={args.n_examples}...")
    print(f"  > Noise Prob: {args.noise_prob}")
    print(f"  > Intensity: {args.feedback_intensity}")
    print(f"  > Folds: {args.n_folds}")
    print(f"  > N Samples (LIME): {args.n_samples}")
    print(f"  > N Features (LIME): {args.n_features}")
    
    # Initialize the exact same problem class as the main experiment
    problem = PROBLEMS[args.problem](
        n_examples=args.n_examples,
        corr_type=args.corr_type,
        n_samples=args.n_samples,
        n_features=args.n_features,
        kernel_width=args.kernel_width,
        lime_repeats=args.lime_repeats,
        vect_type=args.vectorizer,
        rng=rng
    )
    
    print("Executing Passive Evaluation...")
    # This will run the training, then generate corrections, then retrain
    # and save the *_passive_models.pickle file
    eval_passive(problem, args, rng=rng)
    print("Done.")

if __name__ == "__main__":
    run_caipi_baseline()
