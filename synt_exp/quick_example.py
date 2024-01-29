import numpy as np
from smt.qsft import QSFT
from synt_exp.synt_src.synthetic_signal import get_random_subsampled_signal
from smt.random_group_testing import test_uniformity, random_deg_t_vecs, decode, decode_robust
if __name__ == '__main__':
    np.random.seed(8)  # Make it reproducible
    q = 2  # Aspirational
    parameter_set = 1  # Choose which set of parameters to use [1,2,3] (See options below)

    if parameter_set == 1:  # Noiseless Low-Degree
        n = 500
        sparsity = 500
        a_min = -1
        a_max = 1
        b = 9
        noise_sd = 0
        num_subsample = 3
        num_repeat = 1
        wt = 0.9
        p = 200
        t = 10
        noise_model = None
        delays_method_source = "coded"  # Group testing for delays
        delays_method_channel = "identity"
        query_method = "group_testing"  # Group testing for sampling matrix
        source_decoder = decode
        query_args = {
            "query_method": query_method,
            "num_subsample": num_subsample,
            "delays_method_source": delays_method_source,
            "subsampling_method": "smt",
            "delays_method_channel": delays_method_channel,
            "num_repeat": num_repeat,
            "b": b,
            "wt": wt,
            "p": p,
            "t": t
        }
    elif parameter_set == 2:  # Noiseless Uniform Support
        n = 100
        sparsity = 500
        a_min = 1
        a_max = 1
        b = 9
        noise_sd = 0
        num_subsample = 3
        num_repeat = 1
        t = None
        delays_method_source = "identity"
        delays_method_channel = "identity"
        query_method = "simple"
        source_decoder = None
        noise_model = None
        query_args = {
            "query_method": query_method,
            "num_subsample": num_subsample,
            "delays_method_source": delays_method_source,
            "subsampling_method": "smt",
            "delays_method_channel": delays_method_channel,
            "num_repeat": num_repeat,
            "b": b,
        }
    elif parameter_set == 3:  # Noisy Low-Degree
        n = 100
        p = 120
        wt = np.log(2)
        sparsity = 500
        a_min = 1
        a_max = 1
        b = 9
        noise_sd = 0.01
        num_subsample = 3
        num_repeat = 1
        t = 4
        norm_factor = 1
        wt = np.log(2)
        p = 120
        noise_model = "iid_spectral"

        def source_decoder(D, y):
            dec, err, decode_success = decode_robust(D, y, norm_factor, solution=None)
            return dec, decode_success
        delays_method_source = "coded"
        delays_method_channel = "nso"  # Does something
        query_method = "group_testing"
        query_args = {
            "query_method": query_method,
            "num_subsample": num_subsample,
            "delays_method_source": delays_method_source,
            "subsampling_method": "smt",
            "delays_method_channel": delays_method_channel,
            "num_repeat": num_repeat,
            "b": b,
            "wt": wt,
            "p": p,
            "t": t
        }

    qsft_args = {
        "num_subsample": num_subsample,
        "num_repeat": num_repeat,
        "reconstruct_method_source": delays_method_source,
        "reconstruct_method_channel": delays_method_channel,
        "b": b,
        "noise_sd": noise_sd,
        "source_decoder": source_decoder
    }
    '''
    Generate a Signal Object
    '''
    test_signal = get_random_subsampled_signal(n=n,
                                               sparsity=sparsity,
                                               a_min=a_min,
                                               a_max=a_max,
                                               noise_sd=noise_sd,
                                               query_args=query_args,
                                               max_weight=t,
                                               noise_model=noise_model)
    if query_method == "group_testing":
        mean_err, cov_err = test_uniformity(test_signal.Ms[0].T, lambda x: random_deg_t_vecs(t, n, x), 50000)
        print(f"Running Uniformity Check")
        print(f"Normalized Mean L2 ={mean_err}\nNormalized Cov L2 = {cov_err}")

    '''
    Create a QSFT instance and perform the transformation
    '''
    sft = QSFT(**qsft_args)
    result = sft.transform(test_signal, verbosity=10, timing_verbose=True, report=True, sort=True)

    '''
    Display the Reported Results
    '''
    gwht = result.get("gwht")
    loc = result.get("locations")
    n_used = result.get("n_samples")
    peeled = result.get("locations")
    avg_hamming_weight = result.get("avg_hamming_weight")
    max_hamming_weight = result.get("max_hamming_weight")

    print("found non-zero indices QSFT: ")
    print(peeled)
    print("True non-zero indices: ")
    print(test_signal.loc.T)
    print("Total samples = ", n_used)
    print("Total sample ratio = ", n_used / q ** n)
    signal_w_diff = test_signal.signal_w.copy()
    for key in gwht.keys():
        signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
    print("NMSE SMT= ",
         np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(test_signal.signal_w.values())) ** 2))
    print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
    print("Max Hamming Weight of Nonzero Locations = ", max_hamming_weight)
