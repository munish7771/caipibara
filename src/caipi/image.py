import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
from os.path import join
from matplotlib.cm import get_cmap
from itertools import product
from skimage.color import gray2rgb, rgb2gray
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm

from . import Problem, PipeStep, densify, vstack, hstack


class ImageProblem(Problem):
    def __init__(self, **kwargs):
        labels = kwargs.pop('labels')
        images = kwargs.pop('images')
        self.class_names = kwargs.pop('class_names')
        n_examples = kwargs.pop('n_examples', None)
        self.lime_repeats = kwargs.pop('lime_repeats', 1)

        if n_examples is not None:
            rng = check_random_state(kwargs.get('rng', None))
            perm = rng.permutation(len(labels))[:n_examples]
            images, labels = images[perm], labels[perm]

        self.y = labels
        self.confounder_masks = kwargs.pop('confounder_masks', None)
        self.images = self._add_confounders(images)
        self.X = np.stack([gray2rgb(image) for image in self.images], 0)

        self.explainable = set(range(len(self.y)))

        super().__init__(**kwargs)

    def _add_confounders(self, images):
        noisy_images = []
        for image, label in zip(images, self.y):
            confounder = self._y_to_confounder(image, label)
            noisy_images.append(np.maximum(image, confounder))
        return np.array(noisy_images, dtype=np.uint8)

    def _y_to_confounder(self, image, label):
        dd = image.shape[-1] // len(self.class_names)
        ys, xs = range(label * dd, (label + 1) * dd), range(dd)
        mask = np.zeros_like(image)
        mask[np.ix_(ys, xs)] = 255
        return mask

    def preproc(self, images):
        return np.array([rgb2gray(image).ravel() for image in images])

    def explain(self, learner, known_examples, i, pred_y, return_segments=False):
        explainer = LimeImageExplainer(verbose=False)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        # NOTE we *oversegment* the image on purpose!
        segmenter = SegmentationAlgorithm('quickshift',
                                          kernel_size=1,
                                          max_dist=4,
                                          ratio=0.1,
                                          sigma=0,
                                          random_seed=0)
        expl = explainer.explain_instance(self.X[i],
                                          top_labels=len(self.class_names),
                                          classifier_fn=learner.predict_proba,
                                          segmentation_fn=segmenter,
                                          model_regressor=local_model,
                                          num_samples=self.n_samples,
                                          num_features=self.n_features,
                                          batch_size=1,
                                          hide_color=False)
        #print(expl.top_labels)
        _, mask = expl.get_image_and_mask(pred_y,
                                          positive_only=False,
                                          num_features=self.n_features,
                                          min_weight=0.01,
                                          hide_rest=False)
        if return_segments:
            return mask, expl.segments
        return mask

    def query_label(self, i):
        return self.y[i]

    @staticmethod
    def _extract_coords(image, mask):
        return {(r, c)
                for r in range(image.shape[0])
                for c in range(image.shape[1])
                if mask[r, c] != 0}

    def query_corrections(self, i, pred_y, pred_mask, X_test, noise_prob=0.0, feedback_intensity=-1):
        true_y = self.y[i]
        if pred_mask is None:
            return set()
        if pred_y != true_y:
            return set()
        if i not in self.explainable:
            return set()

        image = self.images[i]
        if self.confounder_masks is not None:
            conf_mask = self.confounder_masks[i].copy()
        else:
            conf_mask = self._y_to_confounder(image, self.y[i])
        conf_mask[conf_mask == 255] = 2

        conf_coords = self._extract_coords(image, conf_mask)
        pred_coords = self._extract_coords(image, pred_mask)
        fp_coords = conf_coords & pred_coords

        # Noise injection:
        # With prob noise_prob, add some random "good" pixels to the correction set
        # (simulating user saying "this object part is also a confounder")
        if noise_prob > 0 and self.rng.rand() < noise_prob:
             # Candidates: pixels in prediction but NOT in confounder (true object parts)
             candidates = list(pred_coords - conf_coords)
             if candidates:
                 # Add a random subset of candidates (e.g., 10% or just 1)
                 # Let's add 1-5 random pixels
                 n_noise = self.rng.randint(1, min(6, len(candidates)+1))
                 for _ in range(n_noise):
                     fp_coords.add(candidates[self.rng.randint(len(candidates))])

        X_corrections = []
        
        # Determine values to use for replacement
        if feedback_intensity > 0:
            # Generate 'c' random values in valid range (e.g. -128 to 128 or uint8 0-255 range?)
            # Images are float or uint8? _add_confounders says uint8.
            # But the loop uses [-10, 0, 11].
            # Let's fallback to the original 3 plus random others if c > 3
            values = [250, 0, 11]
            while len(values) < feedback_intensity:
                values.append(self.rng.randint(0, 256))
            values = values[:feedback_intensity]
        else:
            values = [250, 0, 11]

        for value in values:
            corr_image = np.array(image, copy=True)
            for r, c in fp_coords:
                print('correcting pixel {},{} for label {}'.format(
                          r, c, true_y))
                corr_image[r, c] = value
            X_corrections.append(gray2rgb(corr_image))
        n_corrections = len(X_corrections)

        if not n_corrections:
            return set()

        X_corrections = np.array(X_corrections)
        y_corrections = np.array([pred_y] * n_corrections, dtype=np.int8)
        extra_examples = set(range(self.X.shape[0],
                                   self.X.shape[0] + n_corrections))

        self.X = vstack([self.X, X_corrections])
        self.y = hstack([self.y, y_corrections])

        return extra_examples

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):
        if eval_examples is None:
            return -1,

        perfs = []
        for i in set(eval_examples) & self.explainable:
            true_y = self.y[i]
            pred_y = learner.predict(densify(self.X[i]))[0]

            image = self.images[i]
            if self.confounder_masks is not None:
                conf_mask = self.confounder_masks[i].copy()
            else:
                conf_mask = self._y_to_confounder(image, true_y)
            conf_mask[conf_mask == 255] = 2

            pred_mask, segments = \
                self.explain(learner, known_examples, i, pred_y,
                             return_segments=True)

            # Compute confounder recall
            conf_coords = self._extract_coords(image, conf_mask)
            pred_coords = self._extract_coords(image, pred_mask)
            perfs.append(len(conf_coords & pred_coords) / len(conf_coords))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_true.png'.format(i),
                           i, true_y, mask=conf_mask)
            self.save_expl(basename + '_{}_{}_expl.png'.format(i, t),
                           i, pred_y, mask=pred_mask)

        return np.mean(perfs, axis=0),

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        pred_perfs = learner.score(self.X[test_examples],
                                   self.y[test_examples]),
        expl_perfs = self._eval_expl(learner,
                                     known_examples,
                                     eval_examples,
                                     t=t, basename=basename)
        return tuple(pred_perfs) + tuple(expl_perfs)

    def save_expl(self, path, i, y, mask=None, segments=None):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        ax.text(0.5, 1.05,
                'true = {} | this = {}'.format(self.y[i], y),
                horizontalalignment='center',
                transform=ax.transAxes)

        cmap = get_cmap('tab20')

        r, c = self.images[i].shape
        if mask is not None:
            image = np.zeros((r, c, 3))
            for r, c in product(range(r), range(c)):
                image[r, c] = cmap((mask[r, c] & 3) / 3)[:3]
        elif segments is not None:
            image = np.zeros((r, c, 3))
            for r, c in product(range(r), range(c)):
                image[r, c] = cmap((segments[r, c] & 15) / 15)[:3]
        else:
            image = self.X[i]
        ax.imshow(image)

        try:
            fig.savefig(path, bbox_inches=0, pad_inches=0)
        except Exception as e:
            print(f"Error saving plot to {path}: {e}")
        plt.close(fig)


def _load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as fp:
        labels = np.frombuffer(fp.read(), dtype=np.uint8, offset=8)

    images_path = join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(images_path, 'rb') as fp:
        images = np.frombuffer(fp.read(), dtype=np.uint8, offset=16)

    return images.reshape(len(labels), 28, 28), labels


class MNISTProblem(ImageProblem):
    def __init__(self, n_examples=None, **kwargs):
        path = join('data', 'mnist')
        tr_images, tr_labels = _load_mnist(path, kind='train')
        ts_images, ts_labels = _load_mnist(path, kind='t10k')
        images = np.vstack((tr_images, ts_images))
        labels = np.hstack((tr_labels, ts_labels))

        CLASS_NAMES = list(map(str, range(10)))

        super().__init__(images=images,
                         labels=labels,
                         class_names=CLASS_NAMES,
                         n_examples=n_examples,
                         **kwargs)


class FashionProblem(ImageProblem):
    def __init__(self, n_examples=None, **kwargs):
        path = join('data', 'fashion')
        tr_images, tr_labels = _load_mnist(path, kind='train')
        ts_images, ts_labels = _load_mnist(path, kind='t10k')
        
        # Determine shades logic similar to baseline
        shades = np.linspace(0, 255, 10).astype(np.uint8)
        rng = check_random_state(kwargs.get('rng', 42))

        # Apply confounders separately to train and test before stacking
        # to match the baseline logic:
        # Train: shade = f(label)
        # Test: shade = random
        
        def apply_decoy(images, labels, mode='train'):
            noisy = []
            masks = []
            for img, lbl in zip(images, labels):
                r, c = img.shape
                # 4x4 patch in random corner
                corners = [
                    (slice(0, 4), slice(0, 4)),
                    (slice(0, 4), slice(c-4, c)),
                    (slice(r-4, r), slice(0, 4)),
                    (slice(r-4, r), slice(c-4, c))
                ]
                loc = corners[rng.randint(4)]
                
                if mode == 'train':
                    val = shades[lbl]
                else:
                    val = rng.randint(0, 256)
                    
                new_img = img.copy()
                new_img[loc] = val
                noisy.append(new_img)
                
                mask = np.zeros_like(img)
                mask[loc] = 255
                masks.append(mask)
            return np.array(noisy), np.array(masks)

        tr_noisy, tr_masks = apply_decoy(tr_images, tr_labels, mode='train')
        ts_noisy, ts_masks = apply_decoy(ts_images, ts_labels, mode='test')

        images = np.vstack((tr_noisy, ts_noisy))
        masks = np.vstack((tr_masks, ts_masks))
        labels = np.hstack((tr_labels, ts_labels))

        CLASS_NAMES = [
            'T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
            'shirt', 'sneaker', 'bag', 'ankle_boots'
        ]

        # Pass already corrupted images, so we disable _add_confounders in this instance 
        # via a temporary override or by passing clean images?
        # ImageProblem calls _add_confounders(images). 
        # If we pass corrupted images to super, and override _add_confounders to identity, it works.
        
        super().__init__(images=images,
                         labels=labels,
                         class_names=CLASS_NAMES,
                         n_examples=n_examples,
                         confounder_masks=masks,
                         **kwargs)

    def _add_confounders(self, images):
        # Already added in init
        return images
