import subprocess
import os
import sys

# Experiment Parameters
# PROBLEMS = ['colors-rule0', 'fashion']
PROBLEMS = ['fashion'] # Using fashion (decoy) to match baseline
LEARNERS = ['mlp'] # Using mlp to match baseline
STRATEGIES = ['random', 'least-confident'] # Simple baseline strategies
NOISE_PROBS = [0.0, 0.1, 0.3]
INTENSITIES = [1, 5, 20, 100] # feedback_intensity values
def run_experiment(mode='full', start_seed=0, count=5):
    os.makedirs('results', exist_ok=True)
    
    # Path to python executable - using the one running this script
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), 'run_caipi.py')

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
                                S_val = '19'
                                n_val = '100'
                                P_val = '0.01'

                            if mode == 'dry-run':
                                T_val = '5'
                                e_val = '5'
                                S_val = '100'
                                n_val = '100'

                            print("Running: {} {} {} | p={} c={} seed={} | mode={}".format(
                                problem, learner, strategy, noise_prob, intensity, seed, mode))

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'half', 'quarter', 'fast', 'dry-run'],
                        help='Experiment scale: full, half, quarter, fast, dry-run')
    parser.add_argument('--seed', type=int, default=0, help='Starting seed')
    parser.add_argument('--count', type=int, default=5, help='Number of seeds to run')
    args = parser.parse_args()
    
    run_experiment(mode=args.mode, start_seed=args.seed, count=args.count)
