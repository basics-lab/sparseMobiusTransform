import numpy as np
import mobiusmodule
from scipy.fft import fft, ifft
from tqdm import tqdm
import pandas as pd

SETTING = 'matching'  # pick from set {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}
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
    if SETTING == 'matching':
        n = 24
    else:
        n = 25

    N = 2 ** n
    nonzero_mobius = []
    nonzero_fourier = []
    top_coefficients = np.zeros((100, 5))
    top_weighted_coefficients = np.zeros((100, 5))
    NMSE_degree_mobius = np.zeros((100, n+1))
    NMSE_degree_fourier = np.zeros((100, n+1))
    sparsity_degree_mobius_fourier = np.zeros((100, n+1))
    fourier_95 = np.zeros(100)
    fourier_99 = np.zeros(100)
    fourier_999 = np.zeros(100)
    top_coefficients_fourier = np.zeros((100, 5))

    a = np.arange(N, dtype=int)[np.newaxis, :]
    b = np.arange(n, dtype=int)[::-1, np.newaxis]
    allocations = np.array(a & 2 ** b > 0, dtype=int)

    for i in tqdm(range(100)):
        with open(f'saved_values/{SETTING}_{i}', 'rb') as f:
            auction_values = np.load(f)

        '''
        Compute Mobius Transform
        '''
        mt = np.copy(auction_values)
        mobiusmodule.mobius(mt)
        nonzero_mobius.append(np.count_nonzero(mt))

        '''
        Find NMSE for top 100, 200, 300, 400, 500 coefficents
        '''
        for count, j in enumerate(range(100, 600, 100)):
            indices = find_top_magnitude_indices(mt, j)
            sparse_mt = np.zeros(N)
            for ind in indices:
                sparse_mt[ind] = mt[ind]
            mobiusmodule.inversemobius(sparse_mt)
            top_coefficients[i, count] = compute_NMSE(auction_values, sparse_mt)

            indices = find_top_weighted_magnitude_indices(allocations, mt, j)
            sparse_mt = np.zeros(N)
            for ind in indices:
                sparse_mt[ind] = mt[ind]
            mobiusmodule.inversemobius(sparse_mt)
            top_weighted_coefficients[i, count] = compute_NMSE(auction_values, sparse_mt)


        '''
        Compute Fourier Transform
        '''
        ft = fft(auction_values)
        nonzero_fourier.append(np.count_nonzero(ft))
        energy_ratios_sorted = np.sort(np.abs(ft) ** 2 / (np.sum(np.abs(ft) ** 2)))[::-1]
        er_cum_sum = np.cumsum(energy_ratios_sorted)

        fourier_95[i] = np.argmax(er_cum_sum > .95) + 1
        fourier_99[i] = np.argmax(er_cum_sum > .99) + 1
        fourier_999[i] = np.argmax(er_cum_sum > .999) + 1

        '''
        Find NMSE for top 100, 200, 300, 400, 500 coefficents
        '''
        for count, j in enumerate(range(100, 600, 100)):
            indices = find_top_magnitude_indices(ft, j)
            sparse_ft = np.zeros(N, dtype=complex)
            for ind in indices:
                sparse_ft[ind] = ft[ind]
            ift = np.real(ifft(sparse_ft))
            top_coefficients_fourier[i, count] = compute_NMSE(auction_values, ift)

        '''
        Evaluate NMSE of low degree coefficients
        '''
        for deg in range(0, n+1):
            indices = find_up_to_degree_d_indices(allocations, deg)
            deg_d_ft = np.where(indices, ft, 0)
            deg_d_mt = np.where(indices, mt, 0)

            mobiusmodule.inversemobius(deg_d_mt)
            NMSE_degree_mobius[i, deg] = compute_NMSE(auction_values, deg_d_mt)
            NMSE_degree_fourier[i, deg] = compute_NMSE(auction_values, np.real(ifft(deg_d_ft)))

            # Take Mobius of degree d signal from fourier, figure out sparsity
            deg_d_signal = np.real(ifft(deg_d_ft))
            deg_d_signal = np.ascontiguousarray(deg_d_signal, dtype=np.float64)
            mobiusmodule.mobius(deg_d_signal)
            sparsity_degree_mobius_fourier[i, deg] = np.count_nonzero(deg_d_signal)


    '''
    Output results (mean + std) in a csv
    '''
    results = {}

    results['nonzero_mobius_mean'] = np.mean(nonzero_mobius)
    results['nonzero_mobius_std'] = np.std(nonzero_mobius)
    results['nonzero_mobius_min'] = np.min(nonzero_mobius)
    results['nonzero_mobius_max'] = np.max(nonzero_mobius)
    results['nonzero_mobius_median'] = np.median(nonzero_mobius)

    results['nonzero_fourier_mean'] = np.mean(nonzero_fourier)
    results['nonzero_fourier_std'] = np.std(nonzero_fourier)
    results['nonzero_fourier_min'] = np.min(nonzero_fourier)
    results['nonzero_fourier_max'] = np.max(nonzero_fourier)
    results['nonzero_fourier_median'] = np.median(nonzero_fourier)

    results['top_coefficients_100_mean'] = np.mean(top_coefficients[:,0], axis=0)
    results['top_coefficients_100_std'] = np.std(top_coefficients[:,0], axis=0)
    results['top_coefficients_100_min'] = np.min(top_coefficients[:,0], axis=0)
    results['top_coefficients_100_max'] = np.max(top_coefficients[:,0], axis=0)
    results['top_coefficients_100_median'] = np.median(top_coefficients[:,0], axis=0)

    results['top_coefficients_200_mean'] = np.mean(top_coefficients[:,1], axis=0)
    results['top_coefficients_200_std'] = np.std(top_coefficients[:,1], axis=0)
    results['top_coefficients_200_min'] = np.min(top_coefficients[:,1], axis=0)
    results['top_coefficients_200_max'] = np.max(top_coefficients[:,1], axis=0)
    results['top_coefficients_200_median'] = np.median(top_coefficients[:,1], axis=0)

    results['top_coefficients_300_mean'] = np.mean(top_coefficients[:,2], axis=0)
    results['top_coefficients_300_std'] = np.std(top_coefficients[:,2], axis=0)
    results['top_coefficients_300_min'] = np.min(top_coefficients[:,2], axis=0)
    results['top_coefficients_300_max'] = np.max(top_coefficients[:,2], axis=0)
    results['top_coefficients_300_median'] = np.median(top_coefficients[:,2], axis=0)

    results['top_coefficients_400_mean'] = np.mean(top_coefficients[:,3], axis=0)
    results['top_coefficients_400_std'] = np.std(top_coefficients[:,3], axis=0)
    results['top_coefficients_400_min'] = np.min(top_coefficients[:,3], axis=0)
    results['top_coefficients_400_max'] = np.max(top_coefficients[:,3], axis=0)
    results['top_coefficients_400_median'] = np.median(top_coefficients[:,3], axis=0)

    results['top_coefficients_500_mean'] = np.mean(top_coefficients[:,4], axis=0)
    results['top_coefficients_500_std'] = np.std(top_coefficients[:,4], axis=0)
    results['top_coefficients_500_min'] = np.min(top_coefficients[:,4], axis=0)
    results['top_coefficients_500_max'] = np.max(top_coefficients[:,4], axis=0)
    results['top_coefficients_500_median'] = np.median(top_coefficients[:,4], axis=0)

    results['top_weighted_coefficients_100_mean'] = np.mean(top_weighted_coefficients[:,0], axis=0)
    results['top_weighted_coefficients_100_std'] = np.std(top_weighted_coefficients[:,0], axis=0)
    results['top_weighted_coefficients_100_min'] = np.min(top_weighted_coefficients[:,0], axis=0)
    results['top_weighted_coefficients_100_max'] = np.max(top_weighted_coefficients[:,0], axis=0)
    results['top_weighted_coefficients_100_median'] = np.median(top_weighted_coefficients[:,0], axis=0)

    results['top_weighted_coefficients_200_mean'] = np.mean(top_weighted_coefficients[:,1], axis=0)
    results['top_weighted_coefficients_200_std'] = np.std(top_weighted_coefficients[:,1], axis=0)
    results['top_weighted_coefficients_200_min'] = np.min(top_weighted_coefficients[:,1], axis=0)
    results['top_weighted_coefficients_200_max'] = np.max(top_weighted_coefficients[:,1], axis=0)
    results['top_weighted_coefficients_200_median'] = np.median(top_weighted_coefficients[:,1], axis=0)

    results['top_weighted_coefficients_300_mean'] = np.mean(top_weighted_coefficients[:,2], axis=0)
    results['top_weighted_coefficients_300_std'] = np.std(top_weighted_coefficients[:,2], axis=0)
    results['top_weighted_coefficients_300_min'] = np.min(top_weighted_coefficients[:,2], axis=0)
    results['top_weighted_coefficients_300_max'] = np.max(top_weighted_coefficients[:,2], axis=0)
    results['top_weighted_coefficients_300_median'] = np.median(top_weighted_coefficients[:,2], axis=0)

    results['top_weighted_coefficients_400_mean'] = np.mean(top_weighted_coefficients[:,3], axis=0)
    results['top_weighted_coefficients_400_std'] = np.std(top_weighted_coefficients[:,3], axis=0)
    results['top_weighted_coefficients_400_min'] = np.min(top_weighted_coefficients[:,3], axis=0)
    results['top_weighted_coefficients_400_max'] = np.max(top_weighted_coefficients[:,3], axis=0)
    results['top_weighted_coefficients_400_median'] = np.median(top_weighted_coefficients[:,3], axis=0)

    results['top_weighted_coefficients_500_mean'] = np.mean(top_weighted_coefficients[:,4], axis=0)
    results['top_weighted_coefficients_500_std'] = np.std(top_weighted_coefficients[:,4], axis=0)
    results['top_weighted_coefficients_500_min'] = np.min(top_weighted_coefficients[:,4], axis=0)
    results['top_weighted_coefficients_500_max'] = np.max(top_weighted_coefficients[:,4], axis=0)
    results['top_weighted_coefficients_500_median'] = np.median(top_weighted_coefficients[:,4], axis=0)

    results['top_coefficients_fourier_100_mean'] = np.mean(top_coefficients_fourier[:,0], axis=0)
    results['top_coefficients_fourier_100_std'] = np.std(top_coefficients_fourier[:,0], axis=0)
    results['top_coefficients_fourier_100_min'] = np.min(top_coefficients_fourier[:,0], axis=0)
    results['top_coefficients_fourier_100_max'] = np.max(top_coefficients_fourier[:,0], axis=0)
    results['top_coefficients_fourier_100_median'] = np.median(top_coefficients_fourier[:,0], axis=0)

    results['top_coefficients_fourier_200_mean'] = np.mean(top_coefficients_fourier[:,1], axis=0)
    results['top_coefficients_fourier_200_std'] = np.std(top_coefficients_fourier[:,1], axis=0)
    results['top_coefficients_fourier_200_min'] = np.min(top_coefficients_fourier[:,1], axis=0)
    results['top_coefficients_fourier_200_max'] = np.max(top_coefficients_fourier[:,1], axis=0)
    results['top_coefficients_fourier_200_median'] = np.median(top_coefficients_fourier[:,1], axis=0)

    results['top_coefficients_fourier_300_mean'] = np.mean(top_coefficients_fourier[:,2], axis=0)
    results['top_coefficients_fourier_300_std'] = np.std(top_coefficients_fourier[:,2], axis=0)
    results['top_coefficients_fourier_300_min'] = np.min(top_coefficients_fourier[:,2], axis=0)
    results['top_coefficients_fourier_300_max'] = np.max(top_coefficients_fourier[:,2], axis=0)
    results['top_coefficients_fourier_300_median'] = np.median(top_coefficients_fourier[:,2], axis=0)

    results['top_coefficients_fourier_400_mean'] = np.mean(top_coefficients_fourier[:,3], axis=0)
    results['top_coefficients_fourier_400_std'] = np.std(top_coefficients_fourier[:,3], axis=0)
    results['top_coefficients_fourier_400_min'] = np.min(top_coefficients_fourier[:,3], axis=0)
    results['top_coefficients_fourier_400_max'] = np.max(top_coefficients_fourier[:,3], axis=0)
    results['top_coefficients_fourier_400_median'] = np.median(top_coefficients_fourier[:,3], axis=0)

    results['top_coefficients_fourier_500_mean'] = np.mean(top_coefficients_fourier[:,4], axis=0)
    results['top_coefficients_fourier_500_std'] = np.std(top_coefficients_fourier[:,4], axis=0)
    results['top_coefficients_fourier_500_min'] = np.min(top_coefficients_fourier[:,4], axis=0)
    results['top_coefficients_fourier_500_max'] = np.max(top_coefficients_fourier[:,4], axis=0)
    results['top_coefficients_fourier_500_median'] = np.median(top_coefficients_fourier[:,4], axis=0)

    results['fourier_95_mean'] = np.mean(fourier_95)
    results['fourier_95_std'] = np.std(fourier_95)
    results['fourier_95_min'] = np.min(fourier_95)
    results['fourier_95_max'] = np.max(fourier_95)
    results['fourier_95_median'] = np.median(fourier_95)

    results['fourier_99_mean'] = np.mean(fourier_99)
    results['fourier_99_std'] = np.std(fourier_99)
    results['fourier_99_min'] = np.min(fourier_99)
    results['fourier_99_max'] = np.max(fourier_99)
    results['fourier_99_median'] = np.median(fourier_99)

    results['fourier_999_mean'] = np.mean(fourier_999)
    results['fourier_999_std'] = np.std(fourier_999)
    results['fourier_999_min'] = np.min(fourier_999)
    results['fourier_999_max'] = np.max(fourier_999)
    results['fourier_999_median'] = np.median(fourier_999)

    for deg in range(0, n+1):
        results[f'NMSE_degree_mobius_{deg}_mean'] = np.mean(NMSE_degree_mobius[:, deg])
        results[f'NMSE_degree_mobius_{deg}_std'] = np.std(NMSE_degree_mobius[:, deg])
        results[f'NMSE_degree_mobius_{deg}_min'] = np.min(NMSE_degree_mobius[:, deg])
        results[f'NMSE_degree_mobius_{deg}_max'] = np.max(NMSE_degree_mobius[:, deg])
        results[f'NMSE_degree_mobius_{deg}_median'] = np.median(NMSE_degree_mobius[:, deg])

        results[f'NMSE_degree_fourier_{deg}_mean'] = np.mean(NMSE_degree_fourier[:, deg])
        results[f'NMSE_degree_fourier_{deg}_std'] = np.std(NMSE_degree_fourier[:, deg])
        results[f'NMSE_degree_fourier_{deg}_min'] = np.min(NMSE_degree_fourier[:, deg])
        results[f'NMSE_degree_fourier_{deg}_max'] = np.max(NMSE_degree_fourier[:, deg])
        results[f'NMSE_degree_fourier_{deg}_median'] = np.median(NMSE_degree_fourier[:, deg])

        results[f'sparsity_degree_mobius_fourier_{deg}_mean'] = np.mean(sparsity_degree_mobius_fourier[:, deg])
        results[f'sparsity_degree_mobius_fourier_{deg}_std'] = np.std(sparsity_degree_mobius_fourier[:, deg])
        results[f'sparsity_degree_mobius_fourier_{deg}_min'] = np.min(sparsity_degree_mobius_fourier[:, deg])
        results[f'sparsity_degree_mobius_fourier_{deg}_max'] = np.max(sparsity_degree_mobius_fourier[:, deg])
        results[f'sparsity_degree_mobius_fourier_{deg}_median'] = np.median(sparsity_degree_mobius_fourier[:, deg])

    with open(f'results/{SETTING}.csv', 'w') as f:
        results = {k: [v] for k, v in results.items()}
        results_df = pd.DataFrame(results)
        results_df.to_csv(f)
