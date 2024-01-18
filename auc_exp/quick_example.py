import numpy as np
from auc_src.auction_signal_subsampled import AuctionSubsampledSignal
from smt.qsft import QSFT
from smt.query import get_reed_solomon_dec
from synt_exp.synt_src.synthetic_signal import get_random_subsampled_signal

SEED = 0
SETTING = 'arbitrary'  # pick from set {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}


if __name__ == '__main__':

    np.random.seed(8)
    q = 2
    n = 100
    N = q ** n
    sparsity = 5
    a_min = 1
    a_max = 1
    b = 4
    noise_sd = 0
    num_subsample = 3
    num_repeat = 1
    wt = 8
    p = 30
    t = 3
    delays_method_source = "coded"
    delays_method_channel = "identity"

    query_args = {
        "query_method": "group_testing",
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
    }
    '''
    Generate a Signal Object
    '''
    auction_signal = AuctionSubsampledSignal(setting=SETTING, seed=SEED)

    '''
    Create a QSFT instance and perform the transformation
    '''
    sft = QSFT(**qsft_args)
    result = sft.transform(auction_signal, verbosity=10, timing_verbose=True, report=True, sort=True)

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
    print(auction_signal.loc.T)
    print("Total samples = ", n_used)
    print("Total sample ratio = ", n_used / q ** n)
    signal_w_diff = auction_signal.signal_w.copy()
    for key in gwht.keys():
        signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
    print("NMSE SMT= ",
         np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(auction_signal.signal_w.values())) ** 2))
    print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
    print("Max Hamming Weight of Nonzero Locations = ", max_hamming_weight)