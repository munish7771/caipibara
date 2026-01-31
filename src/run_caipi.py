#!/usr/bin/env python3

import sys
import os
from os.path import join, dirname, basename as get_basename


import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join
import os

from caipi import *


PROBLEMS = {
    'toy-fst': lambda *args, **kwargs: \
            ToyProblem(*args, rule='fst', **kwargs),
    'toy-lst': lambda *args, **kwargs: \
            ToyProblem(*args, rule='lst', **kwargs),
    'colors-rule0': lambda *args, **kwargs: \
            ColorsProblem(*args, rule=0, **kwargs),
    'colors-rule1': lambda *args, **kwargs: \
            ColorsProblem(*args, rule=1, **kwargs),
    'reviews': ReviewsProblem,
    'newsgroups': NewsgroupsProblem,
    'fashion': FashionProblem,
}


LEARNERS = {
    'lr': lambda *args, **kwargs: \
            LinearLearner(*args, model='lr', **kwargs),
    'svm': lambda *args, **kwargs: \
            LinearLearner(*args, model='svm', **kwargs),
    'l1svm': lambda *args, **kwargs: \
            LinearLearner(*args, model='l1svm', **kwargs),
    'elastic': lambda *args, **kwargs: \
            LinearLearner(*args, model='elastic', **kwargs),
    'mlp': MLPLearner,
}



def _get_basename(args):
    # Simplified basename to avoid Windows MAX_PATH issues
    basename_str = '__'.join([args.problem, args.learner, args.strategy])
    # key fields only
    fields = [
        ('s', args.seed),
        ('np', args.noise_prob),
        ('fi', args.feedback_intensity),
        ('n', args.n_examples),
    ]
    basename_str += '__' + '__'.join([name + '=' + str(value)
                                  for name, value in fields])
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    return join(output_dir, basename_str)

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = None

    def __enter__(self):
        self.log = open(self.filename, 'w')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        if self.log:
            self.log.close()

    def write(self, message):
         self.terminal.write(message)
         self.log.write(message)
         self.log.flush() # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()



def _subsample(problem, examples, prop, rng=None):
    rng = check_random_state(rng)

    classes = sorted(set(problem.y))
    if 0 <= prop <= 1:
        n_sampled = int(round(len(examples) * prop))
        n_sampled_per_class = max(n_sampled // len(classes), 3)
    else:
        n_sampled_per_class = max(int(prop), 3)

    sample = []
    for y in classes:
        examples_y = np.array([i for i in examples if problem.y[i] == y])
        pi = rng.permutation(len(examples_y))
        sample.extend(examples_y[pi[:n_sampled_per_class]])

    return list(sample)


def eval_passive(problem, args, rng=None):
    """Useful for checking the based performance of the learner and whether
    the explanations are stable."""

    rng = check_random_state(rng)
    basename = _get_basename(args)
    # Plot directory setup
    plot_dir = join('src', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    # plot_basename needs to be in src/plots/ but share the filename part of basename
    plot_basename = join(plot_dir, os.path.basename(basename))


    folds = StratifiedShuffleSplit(n_splits=args.n_folds, random_state=0) \
                .split(problem.y, problem.y)
    train_examples, test_examples = list(folds)[0]
    eval_examples = _subsample(problem, test_examples,
                               args.prop_eval, rng=0)
    print('#train={} #test={} #eval={}'.format(
        len(train_examples), len(test_examples), len(eval_examples)))

    print('  #explainable in train', len(set(train_examples) & problem.explainable))
    print('  #explainable in eval', len(set(eval_examples) & problem.explainable))

    learner = LEARNERS[args.learner](problem, strategy=args.strategy, rng=0)

    learner.fit(problem.X[train_examples],
                problem.y[train_examples])
    train_params = learner.get_params()

    print('Computing full-train performance...')
    perf = problem.eval(learner, train_examples,
                        test_examples, eval_examples,
                        t='train', basename=plot_basename)
    print('perf on full training set =', perf)
    perf_train = perf

    print('Checking LIME stability...')
    perf = problem.eval(learner, train_examples,
                        test_examples, eval_examples,
                        t='train2', basename=plot_basename)
    print('perf on full training set =', perf)

    print('Computing corrections for {} examples...'.format(len(train_examples)))
    X_test_tuples = {tuple(densify(problem.X[i]).ravel())
                     for i in test_examples}

    all_corrections = set()
    for j, i in enumerate(train_examples):
        print('  correcting {:3d} / {:3d}'.format(j + 1, len(train_examples)))
        x = densify(problem.X[i])
        pred_y = learner.predict(x)[0]
        pred_expl = problem.explain(learner, train_examples, i, pred_y)
        corrections = problem.query_corrections(i, pred_y, pred_expl,
                                                X_test_tuples)
        all_corrections.update(corrections)

    print('all_corrections =', all_corrections)

    print('Computing corrected train performance...')
    train_corr_examples = list(sorted(set(train_examples) | all_corrections))
    learner.fit(problem.X[train_corr_examples],
                problem.y[train_corr_examples])
    train_corr_params = learner.get_params()
    perf = problem.eval(learner, train_examples,
                        test_examples, eval_examples,
                        t='train+corr', basename=plot_basename)
    print('perf on corrected set     =', perf)

    print('w_train        :\n', train_params)
    print('w_{train+corr} :\n', train_corr_params)

    dump(basename + '_passive_models.pickle', {
            'w_train': train_params,
            'w_both': train_corr_params,
            'perf_train': perf_train,
            'perf_corrected': perf
        })


def caipi(problem,
          learner,
          train_examples,
          known_examples,
          test_examples,
          eval_examples,
          max_iters=100,
          start_expl_at=-1,
          eval_iters=10,

          noise_prob=0.0,
          feedback_intensity=-1,
          basename=None,
          plot_basename=None,
          snapshot_path=None,
          rng=None):
    rng = check_random_state(rng)

    start_t = 0
    initial_len = len(problem.X)

    if snapshot_path and os.path.exists(snapshot_path):
        print(f"Resuming from snapshot: {snapshot_path}")
        try:
            snap = load(snapshot_path)
            # data integrity check
            if 't' in snap and 'known_examples' in snap:
                 corrections_snap = snap.get('corrections', set())
                 
                 # Restore dynamic data if present
                 if 'X_extra' in snap and len(snap['X_extra']) > 0:
                     problem.X = vstack([problem.X, snap['X_extra']])
                     problem.y = hstack([problem.y, snap['y_extra']])
                 elif corrections_snap and max(corrections_snap) >= initial_len:
                      # If we have indices out of bounds but no data to back them up,
                      # the snapshot is useless for this session.
                      raise ValueError("Snapshot contains corrections but no image data. Cannot resume.")

                 start_t = snap['t'] + 1
                 known_examples = snap['known_examples']
                 corrections = snap['corrections']
                 perfs = snap['perfs']
                 instant_perfs = snap['instant_perfs']
                 params = snap['params']
                 if 'rng_state' in snap:
                     rng.set_state(snap['rng_state'])
                 
                 print(f"  Resumed at iteration {start_t}")
        except Exception as e:
            print(f"  Failed to load snapshot: {e}")
            print("  Starting from scratch.")
            # Ensure we don't start with partial state if exception occurred mid-update
            start_t = 0
            # Note: We rely on the fact that known_examples arg wasn't overwritten if we raised early

    print('CAIPI T={} #train={} #known={} #test={} #eval={}'.format(
          max_iters,
          len(train_examples), len(known_examples),
          len(test_examples), len(eval_examples)))
    print('  #explainable in train', len(set(train_examples) & problem.explainable))
    print('  #explainable in eval', len(set(eval_examples) & problem.explainable))

    X_test_tuples = {tuple(densify(problem.X[i]).ravel())
                     for i in test_examples}

    #learner.select_model(problem.X[train_examples],
    #                     problem.y[train_examples])
    #learner.fit(problem.X[train_examples],
    #            problem.y[train_examples])
    #perf = problem.eval(learner,
    #                    train_examples,
    #                    test_examples,
    #                    eval_examples,
    #                    t='train',
    #                    basename=basename)
    #params = np.round(learner.get_params(), decimals=1)
    #print('train model = {params}, perfs = {perf}'.format(**locals()))

    #learner.select_model(problem.X[known_examples],
    #                     problem.y[known_examples])
    if start_t == 0:
        # Initial fit only if starting from scratch, or re-fit if resuming?
        # If resuming, we need to re-fit with the loaded known_examples + corrections
        # But wait, we haven't loaded them yet if start_t == 0.
        
        # Original code:
        #learner.fit(problem.X[known_examples],
        #            problem.y[known_examples])
        
        corrections = set()
        perfs, instant_perfs, params = [], [], []

    # Always ensure model is fit on current known data before loop
    # If resuming, this restores the model state
    learner.fit(problem.X[known_examples + list(corrections)],
                problem.y[known_examples + list(corrections)])

    for t in range(start_t, max_iters):

        if len(known_examples) >= len(train_examples):
            break

        unknown_examples = set(train_examples) - set(known_examples)
        i = learner.select_query(problem, unknown_examples & problem.explainable)
        assert i in train_examples and i not in known_examples
        x = densify(problem.X[i])

        explain = 0 <= start_expl_at <= t

        pred_y = learner.predict(x)[0]
        pred_expl = problem.explain(learner, known_examples, i, pred_y) \
                    if explain else None

        print('evaluating on query...')
        instant_perf = problem.eval(learner,
                                    known_examples,
                                    [i],
                                    [i],
                                    t=t,
                                    basename=plot_basename + '_instant' if plot_basename else None)
        instant_perfs.append(instant_perf)

        true_y = problem.query_label(i)
        known_examples.append(i)

        if explain:
            new_corrections = problem.query_corrections(i, pred_y, pred_expl,
                                                        X_test_tuples,
                                                        noise_prob=noise_prob,
                                                        feedback_intensity=feedback_intensity)
            corrections.update(new_corrections)

        learner.fit(problem.X[known_examples + list(corrections)],
                    problem.y[known_examples + list(corrections)])
        params.append(learner.get_params())

        do_eval = (eval_iters > 0 and t % eval_iters == 0) or (t == max_iters - 1)

        print('evaluating on test|eval...')
        perf = problem.eval(learner,
                            train_examples,
                            test_examples,
                            eval_examples if do_eval else None,
                            t=t, basename=plot_basename)
        perf = tuple(list(perf) + list([len(corrections)]))
        perfs.append(perf)

        params_for_print = np.round(learner.get_params(), decimals=1)
        print('{t:3d} : model = {params_for_print},  perfs on query = {instant_perf},  perfs on test = {perf}'.format(**locals()))

        if snapshot_path:
             dump(snapshot_path, {
                 't': t,
                 'known_examples': known_examples,
                 'corrections': corrections,
                 'perfs': perfs,
                 'instant_perfs': instant_perfs,
                 'params': params,
                 'rng_state': rng.get_state(),
                 'X_extra': problem.X[initial_len:],
                 'y_extra': problem.y[initial_len:]
             })

    if snapshot_path and os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    return perfs, instant_perfs, params


def eval_interactive(problem, args, rng=None):
    """The main evaluation loop."""

    rng = check_random_state(args.seed)
    basename = _get_basename(args)
    
    # Plot directory setup
    plot_dir = join('src', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_basename = join(plot_dir, os.path.basename(basename))


    folds = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0) \
                .split(problem.y, problem.y)

    perfs, instant_perfs, params = [], [], []

    # Check for existing partial results
    if os.path.exists(basename + '.pickle'):
        try:
            saved_data = load(basename + '.pickle')
            if 'perfs' in saved_data:
                perfs = saved_data['perfs']
                instant_perfs = saved_data['instant_perfs']
                # params might be in a separate file or missing from dictionary in older versions
                # checking param file:
                if os.path.exists(basename + '-params.pickle'):
                    params = load(basename + '-params.pickle')
                
                print(f"Found existing results with {len(perfs)} folds completed.")
        except Exception as e:
            print(f"Error loading existing results: {e}")

    for k, (train_examples, test_examples) in enumerate(folds):
        if k < len(perfs):
            print(f"Skipping fold {k+1}/{args.n_folds} (already done)")
            continue

        print()
        print(80 * '=')
        print('Running fold {}/{}'.format(k + 1, args.n_folds))
        print(80 * '=')

        train_examples = list(train_examples)
        known_examples = _subsample(problem, train_examples,
                                    args.prop_known, rng=0)
        test_examples = list(test_examples)
        eval_examples = _subsample(problem, test_examples,
                                   args.prop_eval, rng=0)

        learner = LEARNERS[args.learner](problem, strategy=args.strategy, rng=0)

        perf, instant_perf, param = \
            caipi(problem,
                  learner,
                  train_examples,
                  known_examples,
                  test_examples,
                  eval_examples,
                  max_iters=args.max_iters,
                  start_expl_at=args.start_expl_at,
                  eval_iters=args.eval_iters,

                  noise_prob=args.noise_prob,
                  feedback_intensity=args.feedback_intensity,
                  basename=basename,
                  plot_basename=plot_basename,
                  snapshot_path=basename + '_fold={}_snapshot.pickle'.format(k),
                  rng=rng)
        perfs.append(perf)
        instant_perfs.append(instant_perf)
        params.append(param)

        dump(basename + '.pickle',
             {'args': args, 'perfs': perfs, 'instant_perfs': instant_perfs})
        dump(basename + '-params.pickle', params)


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('problem', choices=sorted(PROBLEMS.keys()),
                        help='name of the problem')
    parser.add_argument('learner', choices=sorted(LEARNERS.keys()),
                        default='svm', help='Active learner to use')
    parser.add_argument('strategy', type=str, default='random',
                        help='Query selection strategy to use')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-k', '--n-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    group.add_argument('-n', '--n-examples', type=int, default=None,
                       help='Restrict dataset to this many examples')
    group.add_argument('-p', '--prop-known', type=float, default=0.1,
                       help='Proportion of initial labelled examples')
    group.add_argument('-P', '--prop-eval', type=float, default=0.1,
                       help='Proportion of the test set to evaluate the '
                            'explanations on')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='Maximum number of learning iterations')
    group.add_argument('-e', '--eval-iters', type=int, default=10,
                       help='Interval for evaluating performance on the '
                       'evaluation set')
    group.add_argument('--passive', action='store_true',
                       help='DEBUG: eval perfs using passive learning')

    group = parser.add_argument_group('Feedback Noise')
    group.add_argument('--noise-prob', type=float, default=0.0,
                       help='Probability of noisy feedback')
    group.add_argument('--feedback-intensity', type=int, default=-1, # -1 means all
                       help='Number of counterexamples to generate per feedback')

    group = parser.add_argument_group('Interaction')
    group.add_argument('-E', '--start-expl-at', type=int, default=-1,
                       help='Iteration at which corrections kick in')
    group.add_argument('-C', '--corr-type', type=str, default=None,
                       help='Type of correction feedback to use')
    group.add_argument('-F', '--n-features', type=int, default=10,
                       help='Number of LIME features to present the user')
    group.add_argument('-S', '--n-samples', type=int, default=5000,
                       help='Size of the LIME sampled dataset')
    group.add_argument('-K', '--kernel-width', type=float, default=0.75,
                       help='LIME kernel width')
    group.add_argument('-R', '--lime-repeats', type=int, default=1,
                       help='Number of times to re-run LIME')

    group = parser.add_argument_group('Text')
    group.add_argument('--vectorizer', type=str, default=None,
                       help='Text vectorizer to use')
    args = parser.parse_args()

    # np.seterr(all='raise', under='ignore')
    np.set_printoptions(precision=3, linewidth=80, threshold=np.inf)
    np.random.seed(args.seed)

    rng = np.random.RandomState(args.seed)

    print('Creating problem...')
    problem = PROBLEMS[args.problem](n_examples=args.n_examples,
                                     corr_type=args.corr_type,
                                     n_samples=args.n_samples,
                                     n_features=args.n_features,
                                     kernel_width=args.kernel_width,
                                     lime_repeats=args.lime_repeats,
                                     vect_type=args.vectorizer,
                                     rng=rng)

    # Redirect stdout to a log file
    basename = _get_basename(args)
    log_file = basename + '.txt'
    
    print(f"Logging to {log_file}")
    
    with Tee(log_file):
        if args.passive:
            print('Evaluating passive learning...')
            eval_passive(problem, args, rng=rng)
        else:
            print('Evaluating interactive learning...')
            eval_interactive(problem, args, rng=rng)

if __name__ == '__main__':
    main()
