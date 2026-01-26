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
    
    # Path to python executable - using the one running this script
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), 'run_caipi.py')

    planned_tasks = []
    
    # Determine seed range
    if mode == 'fast': count = min(count, 2)
    if mode == 'dry-run': count = 1
    
    end_seed = start_seed + count

    for problem in PROBLEMS:
        for learner in LEARNERS:
            for strategy in STRATEGIES:
                for noise_prob in NOISE_PROBS:
                    for intensity in INTENSITIES:
                        for seed in range(start_seed, end_seed):
                            
                            # Config Logic
                            T_val = '50'
                            e_val = '10'
                            S_val = '1000'
                            n_val = None
                            P_val = '0.1'

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
                                T_val = '20'
                                e_val = '5'
                                S_val = '250'
                                n_val = '100'
                                P_val = '0.01'

                            if mode == 'dry-run':
                                T_val = '5'
                                e_val = '5'
                                S_val = '100'
                                n_val = '100'

                            
                            # Construct expected output filename to check for existence
                            # Format from run_caipi.py: problem__learner__strategy__s=seed__np=noise_prob__fi=intensity__n=n_val
                            n_str = n_val if n_val else 'None' # wait, if n_val is None, arg is not passed, but basename logic uses args.n_examples which would be None? 
                            # Let's check run_caipi.py logic again.
                            # fields = [('s', args.seed), ('np', args.noise_prob), ('fi', args.feedback_intensity), ('n', args.n_examples)]
                            # If n_examples is None, str(None) is 'None'.
                            
                            # However, our current files have n=2000. 
                            # Let's verify how n_val is passed. 
                            # if n_val: cmd.extend(['-n', n_val])
                            # So if n_val is not None, it is passed.
                            
                            # Replicate basename logic
                            basename_n = n_val if n_val else 'None'
                            basename = '{}__{}__{}__s={}__np={}__fi={}__n={}'.format(
                                problem, learner, strategy, seed, noise_prob, intensity, basename_n)
                            # Actually there is more to it, fields are joined by __.
                            # And there are double underscores between main parts and params.
                            # Let's look at the existing file: fashion__mlp__random__s=0__np=0.0__fi=100__n=2000.pickle
                            
                            expected_file = os.path.join('results', basename + '.pickle')
                            
                            import pickle
                            status = "New"
                            details = ""

                            if os.path.exists(expected_file):
                                try:
                                    with open(expected_file, 'rb') as f:
                                        data = pickle.load(f)
                                        # Check if run is complete
                                        saved_args = data.get('args')
                                        perfs = data.get('perfs', [])
                                        
                                        if saved_args and len(perfs) >= saved_args.n_folds:
                                            status = "Done"
                                            # print(f"Skipping completed: {expected_file}", flush=True)
                                            # continue
                                        else:
                                            status = "Resume" 
                                            details = f"({len(perfs)}/{saved_args.n_folds if saved_args else '?'})"
                                            # print(f"Resuming incomplete run: {expected_file} ({len(perfs)}/{saved_args.n_folds if saved_args else '?'})", flush=True)
                                except Exception as e:
                                    status = "Error"
                                    details = str(e)
                                    # print(f"Resuming corrupted/incomplete: {expected_file} (Error: {e})", flush=True)
                            
                            if explain:
                                planned_tasks.append({
                                    'problem': problem,
                                    'learner': learner,
                                    'strategy': strategy,
                                    'noise': noise_prob,
                                    'intensity': intensity,
                                    'seed': seed,
                                    'status': status,
                                    'details': details
                                })
                                continue

                            if status == "Done":
                                print(f"Skipping completed: {expected_file}", flush=True)
                                continue
                            elif status == "Resume":
                                print(f"Resuming incomplete run: {expected_file} {details}", flush=True)
                            elif status == "Error":
                                print(f"Resuming corrupted/incomplete: {expected_file} (Error: {details})", flush=True)


                            print("Running: {} {} {} | p={} c={} seed={} | mode={}".format(
                                problem, learner, strategy, noise_prob, intensity, seed, mode), flush=True)

                            cmd = [
                                python_exe, script_path,
                                problem, learner, strategy,
                                '--noise-prob', str(noise_prob),

                                '--feedback-intensity', str(intensity),
                                '-s', str(seed),
                                '-T', T_val, 
                                '-e', e_val, 
                                '-S', S_val, 
                                '-P', P_val, 
                            ]
                            
                            if n_val:
                                cmd.extend(['-n', n_val])
                            
                            try:
                                subprocess.run(cmd, check=True)
                            except subprocess.CalledProcessError as e:
                                print("Error running command: {}".format(e))
                            
                            if mode == 'dry-run': return

    if explain:
        print(f"\nExperiment Plan (Mode: {mode}, Count: {count})")
        print(f"{'Problem':<15} {'Learner':<10} {'Strategy':<15} {'Noise':<5} {'Int':<5} {'Seed':<5} {'Status':<10} {'Details'}")
        print("-" * 90)
        for t in planned_tasks:
            print(f"{t['problem']:<15} {t['learner']:<10} {t['strategy']:<15} {t['noise']:<5} {t['intensity']:<5} {t['seed']:<5} {t['status']:<10} {t['details']}")
        print("-" * 90)
        
        n_total = len(planned_tasks)
        n_done = sum(1 for t in planned_tasks if t['status'] == 'Done')
        n_new = sum(1 for t in planned_tasks if t['status'] == 'New')
        n_resume = sum(1 for t in planned_tasks if t['status'] == 'Resume')
        
        print(f"Total: {n_total} | Done: {n_done} | New: {n_new} | Resume: {n_resume}")


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
