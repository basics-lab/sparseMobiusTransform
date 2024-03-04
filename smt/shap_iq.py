import time
import numpy as np
from shapiqMain.approximators import SHAPIQEstimator

def shap_iq_decode(signal, n_samples, noise_sd = 0, refine=True, verbose=False, report=True):
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
    t = np.max(np.sum(signal.loc, axis=0))
    dtype = int if (q ** 2)*n > 255 else np.uint8

    start_time = time.time()
    if verbose:
        print("Setting up SHAP IQ problem")

    shapiq_FSI = SHAPIQEstimator(
        interaction_type="FSI", N=range(n), order=t, top_order=True
    )

    shap_start = time.time()

    FSI_scores = shapiq_FSI.compute_interactions_from_budget(
        game=signal.subsampleShapIQ, budget=n_samples,
        pairing=False, sampling_kernel="ksh", only_sampling=False, only_expicit=False, stratification=False
    )

    runtime = time.time() - shap_start

    # Find average time to take one sample, and subtract from SHAP IQ runtime for fair comparison to SMT and LASSO
    sample_time = []
    for i in range(1000):
        subset = []
        rand_sel = np.random.randint(2, size=n)
        for j in range(n):
            if rand_sel[j] == 1:
                subset.append(j)
        time_a = time.time()
        signal.subsampleShapIQ(tuple(subset))
        sample_time.append(time.time() - time_a)

    if n >= 10:
        runtime -= np.mean(sample_time) * n_samples

    gwht_dict = {}
    hamming_weights = []
    for k in FSI_scores.keys():
        nzs = np.nonzero(FSI_scores[k])
        for j in range(len(nzs[0])):
            hamming_weights.append(k)
            ind = []

            for l in range(k):
                ind.append(nzs[l][j])
            tup_ind = []
            for i in range(n):
                if i in ind:
                    tup_ind.append(1)
                else:
                    tup_ind.append(0)
            val = FSI_scores[k][tuple(ind)]
            gwht_dict[tuple(tup_ind)] = val

    if not report:
        return gwht_dict
    else:
        if len(gwht_dict) > 0:
            loc = []
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
