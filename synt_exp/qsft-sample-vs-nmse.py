#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import uuid

sys.path.append("..")
sys.path.append("../src")

from smt.utils import best_convex_underestimator
import argparse
from pathlib import Path
from synt_exp.synt_src.synthetic_helper import SyntheticHelper
from smt.parallel_tests import run_tests

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--num_subsample', type=int, nargs="+")
parser.add_argument('--num_repeat', type=int, nargs="+")
parser.add_argument('--b', type=int, nargs="+")
parser.add_argument('--a', type=int)
parser.add_argument('--noise_sd', type=float, nargs="+")
parser.add_argument('--n', type=int)
parser.add_argument('--q', type=int)
parser.add_argument('--sparsity', type=int)
parser.add_argument('--iters', type=int, default=1)
parser.add_argument('--subsampling', type=int, default=True)
parser.add_argument('--jobid', type=int)

args = parser.parse_args()
debug = args.debug
if debug:
    args.num_subsample = [3]
    args.num_repeat = [2]
    args.b = [4]
    args.a = 1
    args.n = 6
    args.q = 4
    args.sparsity = 1
    args.noise_sd = [1e-3]
    args.iters = 1
    args.jobid = "debug-" + str(uuid.uuid1())[:8]
    args.subsampling = True

if debug:
    exp_dir = Path(f"results/{str(args.jobid)}")
else:
    exp_dir = Path(f"/global/scratch/users/erginbas/smt/synt-exp-results/{str(args.jobid)}")

print("Parameters :", args, flush=True)

query_args = {
    "query_method": "complex",
    "delays_method": "nso",
    "num_subsample": max(args.num_subsample),
    "num_repeat": max(args.num_repeat),
    "b": max(args.b),
    "all_bs": args.b
}

methods = ["smt", "lasso"]
colors = ["red", "blue"]

test_args = {
    "n_samples": 50000
}

print("Loading/Calculating data...", flush=True)

exp_dir.mkdir(parents=True, exist_ok=True)
(exp_dir / "figs").mkdir(exist_ok=True)

helper = SyntheticHelper(args.n, args.q, noise_sd=args.noise_sd[0], sparsity=args.sparsity,
                         a_min=args.a, a_max=args.a,
                         baseline_methods=methods, subsampling=args.subsampling,
                         exp_dir=exp_dir, query_args=query_args, test_args=test_args)

print("n = {}, N = {:.2e}".format(args.n, args.q ** args.n))

print("Starting the tests...", flush=True)

fig, ax = plt.subplots()

for m in range(len(methods)):
    # Test QSFT with different parameters
    # Construct a grid of parameters. For each entry, run multiple test rounds.
    # Compute the average for each parameter selection.
    results_df = run_tests(methods[m], helper, args.iters, args.num_subsample, args.num_repeat,
                           args.b, args.noise_sd, parallel=False)

    # results_df.to_csv(f'results/{str(args.jobid)}/results_df_{methods[m]}.csv')

    means = results_df.groupby(["num_subsample", "num_repeat", "b", "noise_sd"], as_index=False).mean()
    stds = results_df.groupby(["num_subsample", "num_repeat", "b", "noise_sd"], as_index=False).std()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(results_df)

    x_values = []
    y_values = []

    labels = []
    all_points = []

    for i in means.index:
        mean_row = means.iloc[i]
        std_row = stds.iloc[i]
        ax.errorbar(mean_row['n_samples'], mean_row['nmse'],
                    xerr=std_row['n_samples'], yerr=std_row['nmse'], fmt="o", color=colors[m])
        all_points.append([mean_row['n_samples'], mean_row['nmse']])
        label = f'({int(mean_row["b"])},{int(mean_row["num_subsample"])},{int(mean_row["num_repeat"])})'
        labels.append(label)

    for i in range(len(all_points)):
        ax.annotate(labels[i], xy=all_points[i], xycoords='data',
                    xytext=(20, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    shrinkA=0, shrinkB=5,
                                    connectionstyle="arc3,rad=0.4",
                                    color='blue'), )

    try:
        if len(all_points) > 3:
            bcue = best_convex_underestimator(np.array(all_points))
            ax.plot(bcue[:, 0], bcue[:, 1], 'r--', lw=1.5, label="Best Cvx Underest.")
    except:
        pass

ax.set_xlabel('Total Samples')
ax.set_ylabel('Test NMSE')
plt.legend()
plt.grid()
plt.savefig(exp_dir / f'figs/nmse-vs-sample.png')
plt.show()

