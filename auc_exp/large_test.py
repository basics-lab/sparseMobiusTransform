import sys
import os
sys.path.extend(['/global/home/users/landonb/valueFunction'])

import numpy as np
from auc_src.auction_signal_subsampled import AuctionSubsampledSignal
from smt.qsft import QSFT
from smt.utils import bin_vec_to_dec, dec_to_bin_vec
from math import floor
from smt.random_group_testing import decode
import pickle

def random_xor_worst_case_vecs(b, n, num):
    '''
    Generate a random vector of length n with t ones
    '''

    on = np.random.randint(2, size=(b, num))
    vec = np.zeros((n,num))
    vec[:b,:] = on
    return vec

def reconstruction_error(signal, allocations, gwht):
    query_indices_batch = np.array(dec_to_bin_vec(allocations, n)).T
    loc = np.array([np.array(coord) for coord in gwht.keys()]).T
    strengths = np.array(list(gwht.values()))
    if len(gwht.keys()) == 0:
        signal_recon = np.zeros(400)
    else:
        signal_recon = ((((1 - query_indices_batch) @ loc) == 0) + 0) @ strengths
    true_signal = signal.sampling_function(allocations)
    return compute_NMSE(true_signal, signal_recon)

def compute_NMSE(true_signal, recovered_signal):
    '''
    Compute the normalized mean squared error between the true signal and the recovered signal
    '''
    if np.sum(true_signal) == 0:
        if np.sum(recovered_signal) == 0:
            return 0.0
        return 1.0
    else:
        return np.sum((true_signal - recovered_signal) ** 2) / np.sum(true_signal ** 2)

if __name__ == '__main__':

    SETTING = sys.argv[1]
    START_SEED = int(sys.argv[2])
    t_dict = {'arbitrary': 20, 'matching': 4, 'paths': 20, 'proximity': 20, 'scheduling': 8}
    t = t_dict[SETTING]
    b_min = 3
    b_max = 9
    num_subsample_min = 4
    num_subsample_max = 13

    runtimes = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))
    nonzeros_found = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))
    samples_needed = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))
    sampling_ratios = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))
    avg_hamming_weights = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))
    max_hamming_weights = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))
    successful_reconstruction = np.zeros((100, b_max - b_min + 1, num_subsample_max - num_subsample_min + 1))

    n = 400
    N = 2 ** n
    p = 300
    wt = 1.2
    rts = []
    lens = []

    for SEED in range(START_SEED, START_SEED + 5):
        for ind_b, b in enumerate(range(b_min, b_max + 1)):
            for ind_ns, num_subsample in enumerate(range(num_subsample_min, num_subsample_max + 1)):
                print(f"SEED: {SEED}, b: {b}, num_subsample: {num_subsample}")
                '''
                Generate a Signal Object
                '''
                noise_sd = 0
                num_repeat = 1
                noise_model = None
                delays_method_source = "coded"
                delays_method_channel = "identity"
                query_method = "group_testing"
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

                auction_signal = AuctionSubsampledSignal(setting=SETTING, seed=SEED, query_args=query_args)
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
                Create a SMT instance and perform the transformation
                '''
                smt = QSFT(**qsft_args)
                result = smt.transform(auction_signal, verbosity=0, timing_verbose=False, report=True, sort=True)

                '''
                Get SMT Results
                '''
                gwht = result.get("gwht")
                loc = result.get("locations")

                runtimes[SEED, ind_b, ind_ns] = result['runtime']
                nonzeros_found[SEED, ind_b, ind_ns] = len(gwht)
                samples_needed[SEED, ind_b, ind_ns] = result.get("n_samples")
                sampling_ratios[SEED, ind_b, ind_ns] = result.get("n_samples") / 2 ** n
                avg_hamming_weights[SEED, ind_b, ind_ns] = result.get("avg_hamming_weight")
                max_hamming_weights[SEED, ind_b, ind_ns] = result.get("max_hamming_weight")

                # Random allocation with Bernoulli 0.5, most of the allocations will take value 0
                rand_allocs = [floor(np.random.uniform(0, 1) * (2 ** n)) for _ in range(400)]
                NMSE_random = reconstruction_error(auction_signal, rand_allocs, gwht)

                # Allocations where items are added one at a time
                stairstep_allocs = []
                for i in range(n):
                    if i == 0:
                        stairstep_allocs.append(2 ** (n-1))
                    else:
                        st = stairstep_allocs[-1]
                        stairstep_allocs.append(st + 2 ** (n-(i+1)))
                NMSE_stairstep = reconstruction_error(auction_signal, stairstep_allocs, gwht)

                # Random allocation with Bernoulli 0.8, most of the allocations should take positive value
                heavy_allocs = np.where(np.random.uniform(low=0, high=1, size=((400, n))) <= 0.8, 1, 0)
                heavy_allocs_dec = [bin_vec_to_dec(heavy_allocs[i,:]) for i in range(400)]
                NMSE_heavy = reconstruction_error(auction_signal, heavy_allocs_dec, gwht)

                thresh = 1e-8
                print(NMSE_random, NMSE_stairstep, NMSE_heavy)
                if NMSE_random <= thresh and NMSE_stairstep <= thresh and NMSE_heavy <= thresh:
                    successful_reconstruction[SEED, ind_b, ind_ns] = 1
                    print("SUCCESS!")

                auction_signal.pool.close()
    print(SETTING)

    results_dict = {
        'runtimes': runtimes,
        'nonzeros_found': nonzeros_found,
        'samples_needed': samples_needed,
        'sampling_ratios': sampling_ratios,
        'avg_hamming_weights': avg_hamming_weights,
        'max_hamming_weights': max_hamming_weights,
        'successful_reconstruction': successful_reconstruction
    }

    with open(f'/global/scratch/users/landonb/{SETTING}_{START_SEED}.pkl', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)