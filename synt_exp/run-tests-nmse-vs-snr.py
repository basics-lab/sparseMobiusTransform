import numpy as np
import sys
import pandas as pd
import uuid
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)
sys.path.append("..")
import argparse
from pathlib import Path
from synt_exp.synt_src.synthetic_helper import SyntheticHelper
from smt.parallel_tests import run_tests
from synt_exp.synt_src.synthetic_signal import generate_signal_mobius

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--num_subsample', type=int)
    parser.add_argument('--num_repeat', type=int)
    parser.add_argument('--b', type=int)
    parser.add_argument('--snr', type=float, nargs="+")
    parser.add_argument('--n', type=int)
    parser.add_argument('--t', type=int, nargs="+")
    parser.add_argument('--sparsity', type=int)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--subsampling', type=int, default=True)
    parser.add_argument('--jobid', type=int)

    args = parser.parse_args()
    debug = args.debug
    if debug:
        args.num_subsample = 3
        args.num_repeat = 1
        args.b = 10
        args.n = 500
        args.t = [6, 8, 10, 12]
        args.sparsity = 500
        args.iters = 1
        args.jobid = "debug-" + str(uuid.uuid1())[:8]
        args.subsampling = True

    args.snr = np.linspace(-5, 40, num=50)

    if debug:
        exp_dir_base = Path(f"results/{str(args.jobid)}")
    else:
        exp_dir_base = Path(f"/global/scratch/users/erginbas/smt/synt-exp-results/{str(args.jobid)}")

    exp_dir_base.mkdir(parents=True, exist_ok=True)
    (exp_dir_base / "figs").mkdir(exist_ok=True)

    print("Parameters :", args, flush=True)

    methods = ["smt_robust"]

    dataframes = []

    print(f"Starting the tests with ID {args.jobid}", flush=True)

    p = 400
    wt = np.log(2)

    query_args = {
        "num_subsample": args.num_subsample,
        "num_repeat": args.num_repeat,
        "b": args.b,
        "wt": wt,
        "p": p
    }

    test_args = {
        "n_samples": 50000
    }

    print()
    print("n = {}, N = {:.2e}".format(args.n, 2 ** args.n))

    for t_idx in range(len(args.t)):

        noise_sd = np.sqrt(1 / (10 ** (np.array(args.snr) / 10)))

        print(f"noise_sd = {noise_sd}")

        for it in range(args.iters):
            exp_dir = exp_dir_base / f"t{args.t[t_idx]}_i{it}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            _, loc, strengths = generate_signal_mobius(args.n, args.sparsity, -1, 1, max_weight=args.t[t_idx])

            signal_args = {
                "n": args.n,
                "q": 2,
                "t": args.t[t_idx],
                "loc": loc,
                "noise_sd": noise_sd,
                "strengths": strengths,
                "noise_model": "iid_spectral"
            }

            helper = SyntheticHelper(signal_args=signal_args, methods=methods, subsampling=args.subsampling,
                                     exp_dir=exp_dir, query_args=query_args, test_args=test_args)

            for method in methods:
                run_df = run_tests(method, helper, 1, [args.num_subsample], [args.num_repeat],
                                        [args.b], noise_sd, parallel=False)

                run_df["sparsity"] = args.sparsity
                run_df["t"] = args.t[t_idx]
                dataframes.append(run_df)

    results_df = pd.concat(dataframes, ignore_index=True)
    results_df['snr'] = 10 * np.log10(1 / results_df["noise_sd"] ** 2)

    print(results_df)

    results_df.to_pickle(exp_dir_base / "result.pkl")
