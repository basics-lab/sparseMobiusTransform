import numpy as np
from group_lasso import GroupLasso
from sklearn.linear_model import Ridge
import time
from sklearn.utils._testing import ignore_warnings
from smt.utils import calc_hamming_weight, dec_to_bin_vec, bin_vec_to_dec
from sklearn import linear_model
from tqdm import tqdm


def lasso_decode(signal, n_samples, noise_sd = 0, refine=True, verbose=False, report=True):
    """
    Implements Complex LASSO via Fast Iterative Soft Thresholding (FISTA) with optional Ridge Regression refinement
    Parameters
    ---------
    signal : Signal
    Signal object to be transformed.

    n_samples : int
    number of samples used in computing the transform.

    verbosity : bool
    If True printouts are increased.

    noise_sd : scalar
    Noise standard deviation.

    refine : bool
    If True Ridge Regression refinement is used.

    Returns
    -------
    gwht : dict
    Fourier transform (WHT) of the input signal

    runtime : scalar
    transform time + peeling time.

    locations : list
    List of nonzero indicies in the transform.

    avg_hamming_weight : scalar
    Average hamming wieght of non-zero indicies.

    max_hamming_weight : int
    Max hamming weight among the non-zero indicies.
    """
    q = signal.q
    n = signal.n
    N = q ** n
    dtype = int if (q ** 2)*n > 255 else np.uint8
    t = np.max(np.sum(signal.loc, axis=0))

    start_time = time.time()
    if verbose:
        print("Setting up LASSO problem")

    (sample_idx_dec, y) = list(signal.signal_t.keys()), list(signal.signal_t.values())
    if n_samples > len(y):
        n_samples = len(y)
    sample_idx_dec = sample_idx_dec[:n_samples]
    y = y[:n_samples]

    sample_idx_bin = dec_to_bin_vec(sample_idx_dec, n).T
    # for each sample_idx_bin, find all valid subsets

    weight_less_t_filter = np.array([np.sum(dec_to_bin_vec([i],n)) <= t for i in range(2 ** n)], dtype=bool)
    matrix = np.zeros((n_samples, np.sum(weight_less_t_filter)))
    for i in range(n_samples):
        pos_subsets = np.ones([2] * n)
        for j in range(n):
            if sample_idx_bin[i, j] == 0:
                slicer = [slice(None)] * n
                slicer[j] = 0
                pos_subsets[tuple(slicer)] = 0
        matrix[i, :] = np.flip(pos_subsets.reshape(N))[weight_less_t_filter]

    #  WARNING: ADD NOISE ONLY FOR SYNTHETIC SIGNALS
    """
    if signal.is_synt:
        y += np.random.normal(0, noise_sd / np.sqrt(2), size=(len(y), 2)).view(np.complex).reshape(len(y))
    """
    """
    print('Tuning Alpha')
    ''''''
    train_errors = []
    alphas = [1, 0.1, 0.01, 0.001]
    for alpha in alphas:
        lasso = linear_model.Lasso(alpha=alpha)
        lasso.fit(matrix, y)
        train_errors.append(lasso.score(matrix, y))
    alpha = alphas[np.argmax(train_errors)]
    print(train_errors)

    print('Alpha Tuned')
    print(f'Alpha: {alpha}')
    """
    lasso_start = time.time()

    if verbose:
        print(f"Setup Time:{time.time() - start_time}sec")
        print("Running Iterations...")
        start_time = time.time()

    lasso = linear_model.Lasso(alpha=0.05)
    lasso.fit(matrix, y)

    if verbose:
        print(f"LASSO fit time:{time.time() - start_time}sec")

    w = lasso.coef_

    non_zero_pos = np.nonzero(w)[0]

    gwht_dict = {}

    hamming_weights = []
    for p in non_zero_pos:
        # compute the original position in 2^n
        u, i = np.unique(np.cumsum(weight_less_t_filter), return_index=True)
        first_idx = i[p]
        hamming_weights.append(np.sum(dec_to_bin_vec([first_idx], n)))
        gwht_dict[tuple(dec_to_bin_vec([first_idx], n).flatten())] = w[p]

    runtime = time.time() - lasso_start

    if not report:
        return gwht_dict
    else:
        if len(non_zero_pos) > 0:
            loc = list(non_zero_pos)
            avg_hamming_weight = np.mean(hamming_weights)
            max_hamming_weight = np.max(hamming_weights)
        else:
            loc, avg_hamming_weight, max_hamming_weight = [], 0, 0

        result = {
            "gwht": gwht_dict,
            "n_samples": n_samples,
            "locations": loc,
            "runtime": runtime,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight
        }
        return result
