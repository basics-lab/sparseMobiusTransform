import numpy as np
import mobiusmodule
from scipy.fft import fft, ifft
from tqdm import tqdm
import pandas as pd
import json
import sys

# pick from set {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}
def find_top_magnitude_indices(mt, num_kept):
    '''
    Find the indices of the top kept_amt percent of the magnitudes in mt
    '''
    indices = np.argsort(np.abs(mt))[-num_kept:]
    return indices

def find_top_weighted_magnitude_indices(allocs, mt, num_kept):
    '''
    Find the indices of the top kept_amt percent of the magnitudes in mt
    '''
    weights = 2 ** (allocs.shape[0] - np.sum(allocs, axis=0))
    indices = np.argsort(weights * np.abs(mt))[-num_kept:]
    return indices

def find_up_to_degree_d_indices(allocs, d):
    '''
    Find the indices of up to degree d
    '''
    degrees = np.sum(allocs, axis=0)
    return degrees <= d

def compute_NMSE(true_signal, recovered_signal):
    '''
    Compute the normalized mean squared error between the true signal and the recovered signal
    '''
    return np.sum((true_signal - recovered_signal) ** 2) / np.sum(true_signal ** 2)

if __name__ == '__main__':
    SETTING = sys.argv[1]
    n = 20
    N = 2 ** n
    nonzero_mobius = []

    fourier_99 = []

    a = np.arange(N, dtype=int)[np.newaxis, :]
    b = np.arange(n, dtype=int)[::-1, np.newaxis]
    allocations = np.array(a & 2 ** b > 0, dtype=int)


    for i in tqdm(range(100)):
        with open(f'saved_values2/{SETTING}_{i}', 'rb') as f:
            auction_values = np.load(f)

        '''
        Compute Mobius Transform
        '''
        mt = np.copy(auction_values)
        mobiusmodule.mobius(mt)
        nonzero_mobius.append(np.count_nonzero(mt))
        degrees = []
        for i in np.nonzero(mt)[0]:
            degrees.append(np.sum(allocations[:, i]))
        unique, counts = np.unique(degrees, return_counts=True)
        print(np.asarray((unique, counts)).T)

        '''
        Compute Fourier Transform
        '''
        ft = fft(auction_values)
        energy_ratios_sorted = np.sort(np.abs(ft) ** 2 / (np.sum(np.abs(ft) ** 2)))[::-1]
        er_cum_sum = np.cumsum(energy_ratios_sorted)
        fourier_99.append(np.argmax(er_cum_sum > .99) + 1)


    '''
    Output results (mean + std) in a csv
    '''
    results = {}

    results['nonzero_mobius_mean'] = np.mean(nonzero_mobius)
    results['nonzero_mobius_std'] = np.std(nonzero_mobius)
    results['nonzero_mobius_min'] = np.min(nonzero_mobius)
    results['nonzero_mobius_max'] = np.max(nonzero_mobius)
    results['nonzero_mobius_median'] = np.median(nonzero_mobius)
    results['nonzero_mobius_25'] = np.percentile(nonzero_mobius, 2.5)
    results['nonzero_mobius_975'] = np.percentile(nonzero_mobius, 97.5)

    results['fourier_99_mean'] = np.mean(fourier_99)
    results['fourier_99_std'] = np.std(fourier_99)
    results['fourier_99_min'] = np.min(fourier_99)
    results['fourier_99_max'] = np.max(fourier_99)
    results['fourier_99_median'] = np.median(fourier_99)
    results['fourier_99_25'] = np.percentile(fourier_99, 2.5)
    results['fourier_99_975'] = np.percentile(fourier_99, 97.5)

    results = {k: [float(v)] for k, v in results.items()}

    with open(f'results/{SETTING}.json', "w") as outfile:
        json.dump(results, outfile)

