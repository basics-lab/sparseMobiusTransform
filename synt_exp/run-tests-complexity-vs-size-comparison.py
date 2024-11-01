import numpy as np
import sys
import pandas as pd
import uuid
import sys
import os
cwd = os.getcwd()
print(cwd)
print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/global/home/users/landonb/valueFunction'])
print(sys.path)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)

sys.path.append("..")
sys.path.append("../src")

import argparse
from pathlib import Path
from synt_src.synthetic_helper import SyntheticHelper
from smt.parallel_tests import run_tests
from synt_src.synthetic_signal import generate_signal_mobius


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--num_subsample', type=int, nargs="+")
    parser.add_argument('--num_repeat', type=int, nargs="+")
    parser.add_argument('--b', type=int, nargs="+")
    parser.add_argument('--a', type=int)
    parser.add_argument('--snr', type=float)
    parser.add_argument('--n', type=int, nargs="+")
    parser.add_argument('--q', type=int)
    parser.add_argument('--t', type=int, default=None)
    parser.add_argument('--sparsity', type=int)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--subsampling', type=int, default=True)
    parser.add_argument('--jobid', type=int)

    args = parser.parse_args()
    debug = args.debug
    if debug:
        args.num_subsample = [3]
        args.num_repeat = [1]
        args.b = [7]
        args.a = 1
        args.n = list(range(5,51,5))
        args.q = 2
        args.t = None
        args.sparsity = 10
        args.snr = 0
        args.iters = 10
        args.jobid = "debug-" + "timeComplexUniform"
        args.subsampling = True

    if debug:
        exp_dir_base = Path(f"results/{str(args.jobid)}")
    else:
        exp_dir_base = Path(f"/global/scratch/users/landonb/synt-exp-results/{str(args.jobid)}")

    exp_dir_base.mkdir(parents=True, exist_ok=True)
    (exp_dir_base / "figs").mkdir(exist_ok=True)

    print("Parameters :", args, flush=True)

    methods = ["smt", "lasso"]

    dataframes = []

    print("Starting the tests...", flush=True)

    for n_idx in range(len(args.n)):
        for b_idx in range(len(args.b)):

            n = int(args.n[n_idx])
            b = min(n, int(args.b[b_idx]))

            # noise_sd = np.sqrt((args.sparsity * args.a ** 2) / (10 ** (args.snr / 10)))
            noise_sd = 0

            print(fr"n = {n}, N = {args.q ** n:.2e}, b = {b}, sigma = {noise_sd}")

            # b_valid = [b for b in args.b if b <= n]

            subsampling_args = {
                "num_subsample": max(args.num_subsample),
                "num_repeat": max(args.num_repeat),
                "b": b,
            }

            test_args = {
                "n_samples": 200000
            }
            for it in range(args.iters):
                exp_dir = exp_dir_base / f"n{n}_b{b}_i{it}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                _, loc, strengths = generate_signal_mobius(n=n, sparsity=args.sparsity,
                                                           a_min=-args.a, a_max=args.a, max_weight=args.t)
                signal_args = {
                    "n": n,
                    "q": args.q,
                    "t": args.t,
                    "loc": loc,
                    "strengths": strengths,
                    "noise_sd": noise_sd,
                    "noise_model": "iid_spectral",
                    "p": 200,
                    "wt": 0.9
                }

                helper = SyntheticHelper(signal_args=signal_args, methods=methods, subsampling=args.subsampling,
                                         exp_dir=exp_dir, subsampling_args=subsampling_args, test_args=test_args)

                for method in methods:
                    if method == "lasso" and n > 20:
                        # will not compute in under 10 minutes
                        pass
                    elif method == "smt" and (n // b) < args.num_subsample[0]:
                        pass
                    else:
                        dataframes.append(run_tests(method, helper, 1, args.num_subsample, args.num_repeat,
                                                    [b], [noise_sd], parallel=False))
                        results_df = pd.concat(dataframes, ignore_index=True)
                        results_df.to_pickle(exp_dir_base / "result.pkl")

    results_df = pd.concat(dataframes, ignore_index=True)
    results_df.to_pickle(exp_dir_base / "result.pkl")
    print(results_df)