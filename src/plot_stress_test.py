import pickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import re

def parse_filename_params(filename):
    basename = os.path.basename(filename)
    # Extract noise_prob (np) and feedback_intensity (fi)
    # Format: ...__np=0.1__fi=5...
    np_match = re.search(r'np=([\d\.]+)', basename)
    fi_match = re.search(r'fi=(\d+)', basename)
    
    if np_match and fi_match:
        return float(np_match.group(1)), int(fi_match.group(1))
    return None, None

def plot_stress_test(results_dir='results'):
    pickle_files = glob.glob(os.path.join(results_dir, '*__np=*__fi=*.pickle'))
    pickle_files = [f for f in pickle_files if 'params' not in f and 'passive' not in f]

    if not pickle_files:
        print("No stress test result files found.")
        return

    # Data structure: data[problem][noise_prob][intensity] -> list of final accuracies (over seeds)
    aggregated_acc = {}
    aggregated_expl = {}
    
    problems = set()
    noise_probs = set()
    intensities = set()

    for pfile in pickle_files:
        try:
            with open(pfile, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pfile}: {e}")
            continue

        args = data.get('args')
        if not args: continue

        problem = args.problem
        n_p, f_i = parse_filename_params(pfile)
        
        if n_p is None: continue

        problems.add(problem)
        noise_probs.add(n_p)
        intensities.add(f_i)

        if problem not in aggregated_acc:
            aggregated_acc[problem] = {}
            aggregated_expl[problem] = {}
        
        if n_p not in aggregated_acc[problem]:
            aggregated_acc[problem][n_p] = {}
            aggregated_expl[problem][n_p] = {}

        if f_i not in aggregated_acc[problem][n_p]:
            aggregated_acc[problem][n_p][f_i] = []
            aggregated_expl[problem][n_p][f_i] = []

        # Get final performance (last iteration)
        # perfs: (n_folds, n_iters, n_metrics)
        perfs_arr = np.array(data['perfs'])
        
        # Mean across folds for this file (seed)
        # We want the LAST iteration performance
        final_acc = perfs_arr[:, -1, 0].mean() # Metric 0 = Accuracy
        
        # Metric 1 = Confounder Recall
        # Find the last valid explanation score (!= -1) for each fold
        expl_scores = []
        for fold_idx in range(perfs_arr.shape[0]):
            fold_expls = perfs_arr[fold_idx, :, 1]
            # Search backwards
            valid_score = -1.0
            for score in reversed(fold_expls):
                if score != -1:
                    valid_score = score
                    break
            if valid_score != -1:
                expl_scores.append(valid_score)
        
        if expl_scores:
            final_expl = np.mean(expl_scores)
        else:
            final_expl = np.nan

        aggregated_acc[problem][n_p][f_i].append(final_acc)
        aggregated_expl[problem][n_p][f_i].append(final_expl)

    # Plotting
    sorted_intensities = sorted(list(intensities))
    sorted_noise_probs = sorted(list(noise_probs))

    for problem in problems:
        fig, (ax_acc, ax_expl) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Stress Test Results: {problem.upper()}')

        # Plot Accuracy
        for n_p in sorted_noise_probs:
            means = []
            stds = []
            valid_intensities = []
            
            for f_i in sorted_intensities:
                if f_i in aggregated_acc[problem].get(n_p, {}):
                    values = aggregated_acc[problem][n_p][f_i]
                    if values:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                        valid_intensities.append(f_i)
            
            if means:
                ax_acc.errorbar(valid_intensities, means, yerr=stds, label=f'Noise p={n_p}', marker='o', capsize=5)

        ax_acc.set_title('Final Test Accuracy vs Feedback Intensity')
        ax_acc.set_xlabel('Feedback Intensity (c)')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_xscale('log')
        ax_acc.set_xticks(sorted_intensities)
        ax_acc.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()

        # Plot Explanation Quality
        for n_p in sorted_noise_probs:
            means = []
            stds = []
            valid_intensities = []
            
            for f_i in sorted_intensities:
                if f_i in aggregated_expl[problem].get(n_p, {}):
                    values = aggregated_expl[problem][n_p][f_i]
                    if values:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                        valid_intensities.append(f_i)
            
            if means:
                ax_expl.errorbar(valid_intensities, means, yerr=stds, label=f'Noise p={n_p}', marker='x', linestyle='--', capsize=5)

        ax_expl.set_title('Final Confounder Recall vs Feedback Intensity')
        ax_expl.set_xlabel('Feedback Intensity (c)')
        ax_expl.set_ylabel('Confounder Recall (Lower is Better?)')
        # Note: In CAIPI, high recall means we found the confounders (good explanation or bad model dependency?)
        # Actually, high confounder recall means the model is USING the confounder.
        # So we want LOW confounder recall (unlearning).
        
        ax_expl.set_xscale('log')
        ax_expl.set_xticks(sorted_intensities)
        ax_expl.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_expl.grid(True, alpha=0.3)
        ax_expl.legend()

        plt.tight_layout()
        output_file = os.path.join(results_dir, f'stress_test_{problem}_results.png')
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    plot_stress_test()
