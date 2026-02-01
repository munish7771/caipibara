import pickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from collections import defaultdict

def plot_results(results_dir='results'):
    # Find all result pickles matching the pattern
    pickle_files = glob.glob(os.path.join(results_dir, '*__s=*.pickle'))
    pickle_files = [f for f in pickle_files if 'params' not in f and 'passive' not in f]
    
    if not pickle_files:
        print("No result files found.")
        return

    # Group data by configuration
    grouped_data = defaultdict(list)
    noise_probs = set()
    
    print(f"Found {len(pickle_files)} files. Grouping data...")

    for pfile in pickle_files:
        try:
            with open(pfile, 'rb') as f:
                data = pickle.load(f)
            
            args = data.get('args')
            if not args: continue

            key = (args.noise_prob, args.feedback_intensity)
            grouped_data[key].append(data)
            noise_probs.add(args.noise_prob)
        except Exception as e:
            print(f"Error loading {pfile}: {e}")

    # Load Baseline
    baseline_acc = None
    passive_files = glob.glob(os.path.join(results_dir, '*passive_models.pickle'))
    if passive_files:
        latest_passive = max(passive_files, key=os.path.getmtime)
        print(f"Loading baseline from {latest_passive}...")
        try:
            with open(latest_passive, 'rb') as f:
                 bdata = pickle.load(f)
            # perf_corrected is tuple (acc, recall)
            p_corr = bdata.get('perf_corrected')
            if p_corr:
                if isinstance(p_corr, (list, tuple, np.ndarray)):
                    baseline_acc = p_corr[0]
                else:
                    baseline_acc = p_corr
                print(f"Baseline Accuracy: {baseline_acc}")
        except Exception as e:
            print(f"Failed to load baseline: {e}")

    # Create a figure with Rows = Noise Levels, Cols = 1 (Accuracy Only)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows), constrained_layout=True)
    if n_rows == 1: axes = [axes] # Handle single row case

    print(f"Plotting {len(grouped_data)} configurations across {n_rows} noise levels...")
    
    for row_idx, np_val in enumerate(sorted_nps):
        # Get axis for this row
        ax_acc = axes[row_idx]

        # Title for the row
        ax_acc.set_ylabel(f"Noise p={np_val}", fontsize=14, fontweight='bold', labelpad=20)

        # Plot Baseline if available
        if baseline_acc is not None:
             ax_acc.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, label='Passive Baseline (Ideal)')

        # Filter keys for this noise level
        row_keys = [k for k in grouped_data.keys() if k[0] == np_val]
        # Sort by intensity
        row_keys.sort(key=lambda x: x[1])

        for (n_p, fi_val) in row_keys:
            datalist = grouped_data[(n_p, fi_val)]
            
            all_perfs = []
            
            for d in datalist:
                if 'perfs' in d: all_perfs.extend(d['perfs'])
                    
            if not all_perfs: continue
            
            perfs_arr = np.array(all_perfs)
            
            label = f"Intensity {fi_val}"
            
            # 1. Test Accuracy
            mean_acc = np.mean(perfs_arr[:, :, 0], axis=0)
            iters_test = range(len(mean_acc))
            ax_acc.plot(iters_test, mean_acc, label=label, linewidth=2)
            
        # Row styling
        ax_acc.set_title('Predictive Performance (Test)', fontsize=12, fontweight='bold')
        ax_acc.set_xlabel('Iteration')
        if row_idx == 0:
             ax_acc.set_ylabel(f"Noise p={np_val}\nAccuracy")
        else:
             ax_acc.set_ylabel("Accuracy")
            
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(title="Feedback Intensity", fontsize='small')

    save_path = os.path.join(results_dir, 'faceted_accuracy.png')
    plt.savefig(save_path, dpi=150)
    print(f"accuracy plots saved to {save_path}")

if __name__ == "__main__":
    plot_results()
