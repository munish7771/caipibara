import pickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import warnings
from collections import defaultdict

# Dummy class to allow unpickling of BaselineArgs
class BaselineArgs:
    def __init__(self):
        pass

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
    baseline_recall = None
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
                    if len(p_corr) > 1:
                        baseline_recall = p_corr[1]
                else:
                    baseline_acc = p_corr
                print(f"Baseline Accuracy: {baseline_acc}, Recall: {baseline_recall}")
        except Exception as e:
            print(f"Failed to load baseline: {e}")

    # Sort keys for consistent legend order
    sorted_nps = sorted(list(noise_probs))
    # Exclude Noise p=0.0 from plotting rows if requested (as it's used as invalid/baseline)
    sorted_nps = [np_val for np_val in sorted_nps if np_val != 0.0]
    n_rows = len(sorted_nps)
    
    # Calculate Ideal Active Learning Baseline (Noise=0.0, Intensity=1) for ALL metrics
    ideal_acc_curve = None
    ideal_rec_curve = None
    ideal_gap_curve_data = None # Will store (x_range, y_values) for gap

    ideal_key = (0.0, 1)
    if ideal_key in grouped_data:
        print("Found Active Baseline (Noise=0.0)...")
        idl_dat = grouped_data[ideal_key]
        
        idl_perfs = []
        idl_instants = []
        for d in idl_dat:
            if 'perfs' in d: idl_perfs.extend(d['perfs'])
            if 'instant_perfs' in d: idl_instants.extend(d['instant_perfs'])
        
        if idl_perfs:
            idl_arr = np.array(idl_perfs)
            # 1. Ideal Accuracy
            ideal_acc_curve = np.mean(idl_arr[:, :, 0], axis=0)
            
            # 2. Ideal Recall
            if idl_arr.shape[2] > 1:
                rec_data = idl_arr[:, :, 1].copy()
                rec_data[rec_data == -1] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ideal_rec_curve = np.nanmean(rec_data, axis=0)

            # 3. Ideal Gap
            if idl_instants:
                idl_inst_arr = np.array(idl_instants)
                if len(idl_inst_arr) > 0 and idl_inst_arr.ndim >= 3:
                     m_inst = np.mean(idl_inst_arr[:, :, 0], axis=0)
                     m_test = ideal_acc_curve
                     
                     w = 5
                     if len(m_inst) >= w:
                         # Use valid mode for consistency
                         s_tr = np.convolve(m_inst, np.ones(w)/w, mode='valid')
                         start = w // 2
                         end = start + len(s_tr)
                         if len(m_test) >= end:
                             ideal_gap_curve_data = (range(start, end), s_tr - m_test[start:end])
                         else:
                             length = min(len(s_tr), len(m_test))
                             ideal_gap_curve_data = (range(length), s_tr[:length] - m_test[:length])
            
    # Define plot types to generate: (Suffix, MetricKey, Title, YLabel)
    # MetricKey can be 'acc', 'gap', or 'rec'
    plot_configs = [
        ('accuracy', 'acc', 'Predictive Performance (Test Accuracy)', 'Accuracy'),
        ('gap', 'gap', 'Estimated Generalization Gap (Train - Test)', 'Gap'),
        ('recall', 'rec', 'Confounder Recall', 'Recall')
    ]

    # Explicit Style Mapping for Consistency and Emphasis
    # Intensity -> dict(color, linewidth, alpha, zorder)
    # 1 and 100 are emphasized with thicker lines and distinct colors
    style_map = {
        1:   {'color': '#1f77b4', 'lw': 2.5, 'alpha': 1.0, 'zorder': 10}, # Strong Blue
        5:   {'color': '#ff7f0e', 'lw': 1.5, 'alpha': 0.6, 'zorder': 5},  # Muted Orange
        20:  {'color': '#2ca02c', 'lw': 1.5, 'alpha': 0.6, 'zorder': 5},  # Muted Green
        100: {'color': '#d62728', 'lw': 2.5, 'alpha': 1.0, 'zorder': 10}, # Strong Red
    }

    # Fallback colors for unknown intensities
    fallback_colors = ['purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for suffix, metric_type, title_text, ylabel_text in plot_configs:
        print(f"Generating {suffix} plot...")
        
        # Create figure: Rows = 1, Cols = Noise Levels (Horizontal)
        # Height: 5. Width: 6 * n_rows (wider)
        fig, axes = plt.subplots(1, n_rows, figsize=(6 * n_rows, 5), constrained_layout=True)
        
        # Handle single row/col case
        if n_rows == 1: 
            axes = [axes] 

        for col_idx, np_val in enumerate(sorted_nps):
            ax = axes[col_idx]
            
            # Row Header / Axis Labels
            ax.set_title(f"Noise p={np_val}: {title_text}", fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration (Active Learning Steps)')
            if col_idx == 0:
                ax.set_ylabel(ylabel_text)
            ax.grid(True, alpha=0.3)
            
            # Plot Passive Baseline (Horizontal Lines)
            if metric_type == 'acc':
                if baseline_acc is not None:
                    ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Passive Baseline')
            elif metric_type == 'rec':
                if baseline_recall is not None:
                     print(f"DEBUG: Passive Baseline Recall = {baseline_recall}")
                     ax.axhline(y=baseline_recall, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Passive Baseline')
                else:
                     print("DEBUG: Passive Baseline Recall is None")

            # Plot Active Baseline (Noise=0.0) Comparison
            # Only plot if we are looking at a noise condition (to compare against ideal)
            # or if we want it on all plots. Generally useful for np_val != 0.0.
            if np_val != 0.0:
                if metric_type == 'acc' and ideal_acc_curve is not None:
                    ax.plot(range(len(ideal_acc_curve)), ideal_acc_curve, color='black', linestyle=':', linewidth=2, label='Active Baseline (Noise=0.0)', zorder=20)
                elif metric_type == 'rec' and ideal_rec_curve is not None:
                    print(f"DEBUG: Active Baseline Recall (Mean) = {np.nanmean(ideal_rec_curve)}")
                    print(f"DEBUG: First 10 vals: {ideal_rec_curve[:10]}")
                    
                    # Handle NaNs explicitly like the main curves
                    mask = ~np.isnan(ideal_rec_curve)
                    if np.any(mask):
                        x_vals = np.array(range(len(ideal_rec_curve)))[mask]
                        y_vals = ideal_rec_curve[mask]
                        ax.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=2.5, marker='x', markersize=6, label='Active Baseline (Noise=0.0)', zorder=25)
                    else:
                        print("DEBUG: Active Baseline is ALL NaN - cannot plot")

                elif metric_type == 'gap' and ideal_gap_curve_data is not None:
                    ax.plot(ideal_gap_curve_data[0], ideal_gap_curve_data[1], color='black', linestyle=':', linewidth=2, label='Active Baseline (Noise=0.0)', zorder=20)

            # Filter keys for this noise level (row)
            row_keys = [k for k in grouped_data.keys() if k[0] == np_val]
            row_keys.sort(key=lambda x: x[1]) # Sort by feedback intensity

            for i, (n_p, fi_val) in enumerate(row_keys):
                datalist = grouped_data[(n_p, fi_val)]
                
                all_perfs = []
                all_instant = []
                for d in datalist:
                    if 'perfs' in d: all_perfs.extend(d['perfs'])
                    if 'instant_perfs' in d: all_instant.extend(d['instant_perfs'])
                
                if not all_perfs: continue
                perfs_arr = np.array(all_perfs)
                instant_arr = np.array(all_instant)
                
                # Determine x-axis
                iters = range(perfs_arr.shape[1])
                label = f"Intensity {fi_val}"

                # Determine Style
                if fi_val in style_map:
                    s = style_map[fi_val]
                else:
                    # Circular fallback
                    c = fallback_colors[i % len(fallback_colors)]
                    s = {'color': c, 'lw': 1.5, 'alpha': 0.7, 'zorder': 5}

                # PLOTTING LOGIC PER METRIC
                if metric_type == 'acc':
                    # Mean accuracy across seeds
                    mean_acc = np.mean(perfs_arr[:, :, 0], axis=0)
                    ax.plot(iters, mean_acc, label=label, **s)
                
                elif metric_type == 'gap':
                     # Gap = Smoothed(Instant Train) - Test
                     if len(instant_arr) > 0 and instant_arr.ndim >= 3:
                        mean_instant = np.mean(instant_arr[:, :, 0], axis=0)
                        mean_test = np.mean(perfs_arr[:, :, 0], axis=0)
                        
                        # Smooth instant accuracy using valid mode to avoid edge artifacts
                        window = 5
                        if len(mean_instant) >= window:
                            smoothed_train = np.convolve(mean_instant, np.ones(window)/window, mode='valid')
                            
                            # Align test scores to the smoothed train scores (centered window)
                            start_idx = window // 2
                            end_idx = start_idx + len(smoothed_train)
                            
                            if len(mean_test) >= end_idx:
                                mean_test_sliced = mean_test[start_idx:end_idx]
                                gap = smoothed_train - mean_test_sliced
                                ax.plot(range(start_idx, end_idx), gap, label=label, **s)
                            else:
                                # Fallback if sizes mismatch unexpectedly
                                length = min(len(smoothed_train), len(mean_test))
                                gap = smoothed_train[:length] - mean_test[:length]
                                ax.plot(range(length), gap, label=label, **s)
                        else:
                             # Fallback for very short runs
                             length = min(len(mean_instant), len(mean_test))
                             gap = mean_instant[:length] - mean_test[:length]
                             ax.plot(range(length), gap, label=label, **s)

                elif metric_type == 'rec':
                    if perfs_arr.shape[2] > 1:
                        recall_data = perfs_arr[:, :, 1].copy()
                        recall_data[recall_data == -1] = np.nan
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            mean_rec = np.nanmean(recall_data, axis=0)
                        
                        valid = ~np.isnan(mean_rec)
                        if np.any(valid):
                            ax.plot(np.array(iters)[valid], mean_rec[valid], label=label, marker='o', markersize=4, **s)

            # Legend
            ax.legend(title="Feedback Intensity", fontsize='small', loc='best')
            
            # Limits Use slightly wider limits to ensure Baselines (esp 0.0) are visible
            if metric_type == 'rec':
                ax.set_ylim(0.03, 0.10)

        # Save individual figure
        save_path = os.path.join(results_dir, f'results_{suffix}_horizontal.png')
        fig.savefig(save_path, dpi=120)
        plt.close(fig) # Close to free memory
        print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_results()
