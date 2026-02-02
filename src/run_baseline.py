#!/usr/bin/env python3
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

# Import the necessary components from the main script
from src.run_caipi import PROBLEMS, eval_interactive

class BaselineArgs:
    """Mock arguments object to pass to eval_interactive"""
    def __init__(self):
        # Core Settings
        self.problem = 'fashion'
        self.learner = 'mlp'
        self.strategy = 'least-confident' # Match stress test
        self.seed = 0
        
        # Data Config (Matching Stress Test 'quarter' mode)
        self.n_examples = 2000
        self.n_folds = 10  # Match stress test default
        self.prop_known = 0.05 # Initial known
        self.prop_eval = 0.05
        
        # THE REQUESTED CONFIGURATION
        self.noise_prob = 0.0          # Zero Noise (Ideal)
        self.feedback_intensity = 1    # 1 Counterexample
        
        # Interaction Params
        self.max_iters = 50
        self.eval_iters = 1 # Eval every step for smooth baseline curve? Or 10 to match stress test?
                            # Stress test uses T=50, e=10. Let's use 5 to be slightly smoother or 10.
        self.eval_iters = 5 
        self.start_expl_at = 0
        
        # Problem Params (Defaults)
        self.corr_type = None
        self.n_samples = 200    # Quarter mode: 200
        self.n_features = 3     # Quarter mode: 3
        self.kernel_width = 0.75
        self.lime_repeats = 1
        self.vectorizer = None
        
        self.passive = False # Ensure we don't trigger passive mode logic in logging

def run_caipi_baseline():
    print("--- Running Standardized CAIPI Baseline (Ideal Active Learning) ---")
    print("Configuration: Noise=0.0, Intensity=1, Strategy=Least-Confident, T=50")
    
    args = BaselineArgs()
    rng = np.random.RandomState(args.seed)
    
    print(f"Initializing {args.problem} problem with n={args.n_examples}...")
    print(f"  > Noise Prob: {args.noise_prob}")
    print(f"  > Intensity: {args.feedback_intensity}")
    print(f"  > Folds: {args.n_folds}")
    print(f"  > Max Iters: {args.max_iters}")
    
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
    
    print("Executing Interactive Evaluation...")
    eval_interactive(problem, args, rng=rng)
    print("Done.")

if __name__ == "__main__":
    run_caipi_baseline()
