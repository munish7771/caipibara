import subprocess
import os
import sys

# Experiment Parameters
# PROBLEMS = ['colors-rule0', 'fashion']
PROBLEMS = ['fashion'] # Using fashion (decoy) to match baseline
LEARNERS = ['mlp'] # Using mlp to match baseline
STRATEGIES = ['least-confident'] # Simple baseline strategies
NOISE_PROBS = [0.1, 0.5, 1.0]
INTENSITIES = [1, 5, 20, 100] # feedback_intensity values

def run_experiment(mode='full', start_seed=0, count=5, explain=False):
    os.makedirs('results', exist_ok=True)
    
    # Ensure unbuffered output for subprocesses
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Path to python executable - using the one running this script
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), 'run_caipi.py')

    planned_tasks = []
    
    # Determine seed range
    if mode == 'fast': count = min(count, 2)
    if mode == 'quarter': count = min(count, 2)
    if mode == 'dry-run': count = 1
    
    end_seed = start_seed + count

    # Collect all tasks first
    for problem in PROBLEMS:
        for learner in LEARNERS:
            for strategy in STRATEGIES:
                for noise_prob in NOISE_PROBS:
                    for intensity in INTENSITIES:
                        for seed in range(start_seed, end_seed):
                            planned_tasks.append({
                                'problem': problem,
                                'learner': learner,
                                'strategy': strategy,
                                'noise_prob': noise_prob,
                                'intensity': intensity,
                                'seed': seed
                            })

    if explain:
        print(f"\nExperiment Plan (Mode: {mode}, Count: {count})")
        print(f"{'Problem':<15} {'Learner':<10} {'Strategy':<15} {'Noise':<5} {'Int':<5} {'Seed':<5} {'Status':<10}")
        print("-" * 80)
        
    for i, task in enumerate(planned_tasks):
        problem = task['problem']
        learner = task['learner']
        strategy = task['strategy']
        noise_prob = task['noise_prob']
        intensity = task['intensity']
        seed = task['seed']
        
        # Config Logic
        T_val = '50'
        e_val = '10'
        S_val = '1000'
        n_val = None
        P_val = '0.1'
        F_val = '3' # Reduced from default 10 to ensure valid noise injection checks

        if mode == 'quarter':
            # Fast/Light (~2 hours/seed)
            S_val = '200'
            n_val = '2000'
            P_val = '0.05'
        elif mode == 'half':
            # Medium (~overnight)
            S_val = '500'
            n_val = '10000'
            P_val = '0.1'
        elif mode == 'full':
            # Heavy (~days)
            S_val = '1000'
            n_val = None # All 60k
            P_val = '0.1'
        elif mode == 'fast': # Dev/Debug
            T_val = '5'
            e_val = '5'
            S_val = '100'
            n_val = '50'
            P_val = '0.01'

        if mode == 'dry-run':
            T_val = '5'
            e_val = '5'
            S_val = '100'
            n_val = '100'

        
        # Construct expected output filename to check for existence
        # Replicate basename logic
        basename_n = n_val if n_val else 'None'
        basename = '{}__{}__{}__s={}__np={}__fi={}__n={}'.format(
            problem, learner, strategy, seed, noise_prob, intensity, basename_n)
        
        expected_file = os.path.join('results', basename + '.pickle')
        
        import pickle
        status = "New"

        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'rb') as f:
                    data = pickle.load(f)
                    # Check if run is complete
                    saved_args = data.get('args')
                    perfs = data.get('perfs', [])
                    
                    if saved_args and len(perfs) >= saved_args.n_folds:
                        status = "Done"
                    else:
                        status = "Resume" 
            except Exception as e:
                status = "Error"
        
        if explain:
            print(f"{problem:<15} {learner:<10} {strategy:<15} {noise_prob:<5} {intensity:<5} {seed:<5} {status:<10}")
            continue

        if status == "Done":
             # print(f"Skipping completed: {expected_file}", flush=True)
             continue
        elif status == "Resume":
             print(f"Resuming incomplete run: {expected_file}", flush=True)

        print(f"[{i+1}/{len(planned_tasks)}] Running: {problem} {learner} | p={noise_prob} c={intensity} seed={seed}")

        cmd = [
            python_exe, script_path,
            problem, learner, strategy,
            '--noise-prob', str(noise_prob),
            '--feedback-intensity', str(intensity),
            '-s', str(seed),
            '-T', T_val, 
            '-e', e_val, 
            '-S', S_val, 
            '-S', S_val, 
            '-P', P_val, 
            '-F', F_val,
            '-E', '0', # Enable corrections from start
        ]
        
        if n_val:
            cmd.extend(['-n', n_val])
        
        try:
            # Allow output to show internal progress (LIME bars)
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for {problem} {learner} p={noise_prob} c={intensity}: {e}")
            # Continue to next task instead of crashing
            continue
        
        if mode == 'dry-run': return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'half', 'quarter', 'fast', 'dry-run'],
                        help='Experiment scale: full, half, quarter, fast, dry-run')
    parser.add_argument('--seed', type=int, default=0, help='Starting seed')
    parser.add_argument('--count', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--explain', action='store_true', help='Print experiment plan without running')
    args = parser.parse_args()
    
    run_experiment(mode=args.mode, start_seed=args.seed, count=args.count, explain=args.explain)
