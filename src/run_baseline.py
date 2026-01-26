import numpy as np
import gzip
import os
from os.path import join
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    images_path = join(path, '{}-images-idx3-ubyte.gz'.format(kind))

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"File not found: {labels_path}")
    
    with gzip.open(labels_path, 'rb') as fp:
        labels = np.frombuffer(fp.read(), dtype=np.uint8, offset=8)

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"File not found: {images_path}")

    with gzip.open(images_path, 'rb') as fp:
        images = np.frombuffer(fp.read(), dtype=np.uint8, offset=16)

    return images.reshape(len(labels), 28, 28), labels

def add_confounders(images, labels, mode='train', seed=0):
    rng = np.random.RandomState(seed)
    noisy = []
    
    # 10 classes, map to 10 distinct shades
    # Use linspace to get well-separated values (0, 28, 56, ..., 255)
    shades = np.linspace(0, 255, 10).astype(np.uint8)
    
    for img, lbl in zip(images, labels):
        r, c = img.shape
        # 4x4 patch in random corner
        # Four corners: TL, TR, BL, BR
        corners = [
            (slice(0, 4), slice(0, 4)),
            (slice(0, 4), slice(c-4, c)),
            (slice(r-4, r), slice(0, 4)),
            (slice(r-4, r), slice(c-4, c))
        ]
        loc = corners[rng.randint(4)]
        
        # Shade logic
        if mode == 'train':
            val = shades[lbl]
        else: # test or random (for CE)
            val = rng.randint(0, 256)
            
        new_img = img.copy()
        new_img[loc] = val
        noisy.append(new_img)
        
    return np.array(noisy)

def flatten(images):
    return images.reshape(images.shape[0], -1)

def run_experiment():
    # Attempt to locate data
    # Assuming standard path relative to project root
    # Adjust as necessary
    path = join('data', 'fashion')
    
    print(f"Loading data from {path}...")
    try:
        train_images, train_labels = load_mnist(path, 'train')
        test_images, test_labels = load_mnist(path, 't10k')
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'data/fashion/' contains the MNIST .gz files.")
        return

    print(f"Loaded {len(train_images)} training and {len(test_images)} test images.")

    # --- Baseline: No Corrections ---
    print("\n--- Baseline: No Corrections ---")
    
    # Train set: Confounder shade is function of label
    print("Generating Confounded Training Set (Mode=Train)...")
    X_train_bad = add_confounders(train_images, train_labels, mode='train', seed=42)
    
    # Test set: Confounder shade is random
    print("Generating Confounded Test Set (Mode=Test)...")
    X_test_bad = add_confounders(test_images, test_labels, mode='test', seed=100) 

    # MLP Setup
    # Ross et al 2017: "two hidden layers of 100 units each"
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42, max_iter=200)
    
    print("Training MLP (Baseline)...")
    # Using 28*28 = 784 features
    mlp.fit(flatten(X_train_bad), train_labels)
    acc = mlp.score(flatten(X_test_bad), test_labels)
    print(f"Baseline Accuracy (Expected ~48%): {acc*100:.2f}%")

    # Collect results for plotting
    results = {'Baseline': acc}
    
    # --- Corrections (CE) ---
    # Strategy: Add c counterexamples per training image
    # Reformulate training data: Original (Bad) + c * Augmented (Randomized Confounder)
    
    c_values = [1, 3, 5]
    for c_val in c_values:
        print(f"\n--- CE Strategy: c={c_val} ---")
        
        X_aug_list = []
        y_aug_list = []
        
        print(f"Generating {c_val}x Counterexamples...")
        for k in range(c_val):
            # Use distinct seeds to get different random choices (corners/shades)
            # mode='test' implies random shades, simulating 'decoy pixels randomized'
            # We pass original CLEAN images to add_confounders to generate fresh variants
            X_aug_k = add_confounders(train_images, train_labels, mode='test', seed=1000 + k)
            X_aug_list.append(X_aug_k)
            y_aug_list.append(train_labels) 
            
        X_aug = np.concatenate(X_aug_list, axis=0)
        y_aug = np.concatenate(y_aug_list, axis=0)
        
        # Combine Original (Bad) + Augmented
        X_final = np.concatenate([X_train_bad, X_aug], axis=0)
        y_final = np.concatenate([train_labels, y_aug], axis=0)
        
        print(f"Training MLP on {len(X_final)} examples...")
        mlp_ce = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42, max_iter=200)
        mlp_ce.fit(flatten(X_final), y_final)
        acc_ce = mlp_ce.score(flatten(X_test_bad), test_labels)
        print(f"CE (c={c_val}) Accuracy (Expected >80%): {acc_ce*100:.2f}%")
        results[f'CE (c={c_val})'] = acc_ce

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        names = list(results.keys())
        values = [v * 100 for v in results.values()]
        
        plt.bar(names, values, color=['gray', 'tab:blue', 'tab:blue', 'tab:blue'])
        plt.ylabel('Test Accuracy (%)')
        plt.title('Effect of Counterexamples on Decoy Fashion MNIST')
        plt.ylim(0, 100)
        
        for i, v in enumerate(values):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
            
        output_path = 'rq2_baseline_results.png'
        plt.savefig(output_path)
        print(f"\nPlot saved to {output_path}")
        
    except ImportError:
        print("\nMatplotlib not found. Skipping plotting.")


if __name__ == '__main__':
    run_experiment()
