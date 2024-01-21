import numpy as np
from auc_src.auction_signal_subsampled import AuctionSubsampledSignal
from smt.qsft import QSFT
from smt.fmt import fmt_recursive, ifmt_recursive
from smt.utils import bin_vec_to_dec
import mobiusmodule
import os

SETTING = 'arbitrary'  # pick from set {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}


if __name__ == '__main__':
    if SETTING == 'matching':
        n = 24
    else:
        n = 25
    q = 2
    b = 9
    noise_sd = 0
    num_subsample = 3
    num_repeat = 1
    wt = 8
    p = 30
    t = 25
    delays_method_source = "identity"
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
    auction_signal = AuctionSubsampledSignal(setting=SETTING, seed=SEED, query_args=query_args)
    n = auction_signal.n
    N = 2 ** n

    a = np.arange(N, dtype=int)[np.newaxis, :]
    b = np.arange(n, dtype=int)[::-1, np.newaxis]
    allocations = np.array(a & 2 ** b > 0, dtype=int)

    dir = os.path.dirname(__file__)
    value_filepath = os.path.join(dir, f'saved_values/values_{SETTING}_{SEED}.npy')

    if os.path.isfile(value_filepath):
        with open(value_filepath, 'rb') as f:
            auction_values = np.load(f)
    else:
        auction_values = auction_signal.subsample(range(N))
        with open(value_filepath, 'wb') as f:
            np.save(f, auction_values)

    mt = np.copy(auction_values)
    print(mt)
    mobiusmodule.mobius(mt)
    print(mt)
    mobiusmodule.inversemobius(mt)
    print(mt)
    print(np.sum(auction_values - mt))

    true_nonzero_mt = np.nonzero(mt)[0]
    print(f'Count true nonzero: {len(true_nonzero_mt)}')
    true_mobius_coefs = {}
    # print(true_nonzero_mt)
    for i in true_nonzero_mt:
        true_mobius_coefs[tuple(allocations[:, i])] = mt[i]
    # print('True mobius coefficients')
    # print(true_mobius_coefs)

    '''
    Create a SMT instance and perform the transformation
    '''
    smt = QSFT(**qsft_args)
    result = smt.transform(auction_signal, verbosity=0, timing_verbose=True, report=True, sort=True)

    '''
    Display the Reported Results
    '''
    gwht = result.get("gwht")
    loc = result.get("locations")
    n_used = result.get("n_samples")
    peeled = result.get("locations")
    avg_hamming_weight = result.get("avg_hamming_weight")
    max_hamming_weight = result.get("max_hamming_weight")

    mt_smt = np.zeros(N)
    for key in gwht.keys():
        mt_smt[bin_vec_to_dec(np.array(key))] = gwht[key]

    values_recon = ifmt_recursive(mt_smt)
    print(gwht)
    print('Num nonzero found = ', len(gwht))
    print("Total samples = ", n_used)
    print("Total sample ratio = ", n_used / q ** n)
    print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
    print("Max Hamming Weight of Nonzero Locations = ", max_hamming_weight)

    print("NMSE SMT= ",
          np.sum(np.abs(values_recon - auction_values) ** 2) / np.sum(
              np.abs(auction_values) ** 2))

    auction_signal.pool.close()
