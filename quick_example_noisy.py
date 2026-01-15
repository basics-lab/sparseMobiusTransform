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

    if parameter_set == 1:  # Noisy Low-Degree
        n = 300
        sparsity = 100
        a_min = 1
        a_max = 2
        t = 10
        # SMT parameters
        b_smt = 7
        num_subsample_smt = 3
        num_repeat_smt = 1
        p_smt = 100
        noise_sd = 0.2
    else:
        raise NotImplementedError
    '''
    Generate signal parameters
    '''
    signal_w, signal_loc, signal_strengths = generate_signal_mobius(n, sparsity, a_min, a_max, max_weight=t)

    signal_loc_csc = csc_array(signal_loc)

    def test_function(query_batch):
        output = ((((1 - np.array(query_batch)) @ signal_loc_csc) == 0) + 0) @ signal_strengths
        return output + noise_sd * np.random.normal(size=len(query_batch))

    algos = [("SMT", {"n": n, "b": b_smt, "num_subsample": num_subsample_smt, "num_repeat": num_repeat_smt, "p": p_smt, "t": t})]

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
        print("Total numder of batches = ", n_batches)
        # print(f"Information theoretic sample lower bound (worst case) = {int(np.ceil(sparsity * np.log(binom(n, t)) / np.log(sparsity + 1)))}")

        signal_w_diff = signal_w.copy()
        for key in transform.keys():
            signal_w_diff[key] = signal_w_diff.get(key, 0) - transform[key]
        print(f"NMSE = {np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(signal_w.values())) ** 2)}")
        print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
        print("Max Hamming Weight of Nonzero Locations = ", max_hamming_weight)

        print(f"Total Runtime = {runtime}, Sampling Time = {sampling_time}")
