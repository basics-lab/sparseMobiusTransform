import numpy as np

from scipy.sparse import csc_array

from sparse_transform import run_transforms
from synt.synthetic_signal import generate_signal_mobius

import colorama
import sys
import logging


if __name__ == '__main__':
    # np.random.seed(8)  # Make it reproducible
    q = 2  # Aspirational
    parameter_set = 1  # Choose which set of parameters to use (See options below)

    logging.basicConfig(level=logging.INFO)

    if parameter_set == 1:  # Noiseless Low-Degree
        n = 1000
        sparsity = 100
        a_min = 1
        a_max = 2
        t = 10
        # SMT parameters
        b_smt = 6
        num_subsample_smt = 3
        num_repeat_smt = 1
        p_smt = 200
        # ASMT parameters
        b_asmt = 200
        # Batched FASMT parameters
        b_fasmt = 6
        noise_model = None
    elif parameter_set == 2:  # Noiseless Low-Degree SIMPLE
        n = 20
        b_smt = 3
        num_subsample_smt = 3
        num_repeat_smt = 1
        p_smt = 20
        b_asmt = 30  # for ASMT
        b_fasmt = 2  # for Batched FASMT
        sparsity = 3
        a_min = 1
        a_max = 2
        t = 3  # only for generating the signal
        noise_model = None
    else:
        raise NotImplementedError
    '''
    Generate signal parameters
    '''
    signal_w, signal_loc, signal_strengths = generate_signal_mobius(n, sparsity, a_min, a_max, max_weight=t)

    signal_loc_csc = csc_array(signal_loc)

    def test_function(query_batch):
        return ((((1 - np.array(query_batch)) @ signal_loc_csc) == 0) + 0) @ signal_strengths

    algos = [
        ("SMT", {"n": n, "b": b_smt, "num_subsample": num_subsample_smt, "num_repeat": num_repeat_smt, "p": p_smt, "t": t, "noise_sd": 0}),
        # ("ASMT", {"n": n, "noise_sd": 0}),
        ("ASMT", {"n": n, "b": b_asmt, "t": t, "noise_sd": 0}),
        ("FASMT", {"n": n, "noise_sd": 0}),
        ("FASMTBatched", {"n": n, "b": b_fasmt, "t": t, "noise_sd": 0}),
        ("FASMTParallel", {"n": n, "b": b_fasmt, "t": t, "noise_sd": 0}),
    ]

    '''
    Run each algorithm and analyze the results
    '''
    for algo, config in algos:
        result = run_transforms(test_function, algo, **config)

        sys.stderr.flush()
        sys.stdout.flush()

        print(f'{colorama.Fore.RED}---------------------------------------------\nTest Results for {algo}{colorama.Fore.RESET}')

        '''
        Display the Reported Results
        '''
        transform = result.get("transform")
        loc = result.get("locations")
        n_used = result.get("n_samples")
        n_batches = result.get("n_batches")
        peeled = result.get("locations")
        avg_hamming_weight = result.get("avg_hamming_weight")
        max_hamming_weight = result.get("max_hamming_weight")

        runtime = result.get('runtime')
        sampling_time = result.get('sampling_time')


        def color_sign(x):
            c = colorama.Fore.RED if x > 0 else colorama.Fore.RESET
            return f'{c}{x}{colorama.Fore.RESET}'


        sys.stderr.flush()
        sys.stdout.flush()
        np.set_printoptions(formatter={'int': color_sign}, threshold=1000, linewidth=1000, edgeitems=5)

        print("found non-zero indices QSFT: ")
        print(peeled)
        print("True non-zero indices: ")
        print(signal_loc.T)

        # reset the print options
        np.set_printoptions()

        print("Total samples = ", n_used)
        print("Total sample ratio = ", n_used / q ** n)
        print("Total batches = ", n_batches)
        # print(f"Information theoretic sample lower bound (worst case) = {int(np.ceil(sparsity * np.log(binom(n, t)) / np.log(sparsity + 1)))}")

        signal_w_diff = signal_w.copy()
        for key in transform.keys():
            signal_w_diff[key] = signal_w_diff.get(key, 0) - transform[key]
        print(f"NMSE = {np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(signal_w.values())) ** 2)}")
        print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
        print("MAX Hamming Weight of Nonzero Locations = ", max_hamming_weight)

        print(f"Total Runtime = {runtime}, Sampling Time = {sampling_time}")
