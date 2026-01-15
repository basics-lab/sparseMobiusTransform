'''
Class for computing the q-ary fourier transform of a function/signal
'''
import time
import numpy as np
from sparse_transform.smt.utils import calc_hamming_weight, sort_vecs
from scipy import linalg as la
from sparse_transform.smt.query import get_Ms
from sparse_transform.smt.random_group_testing import test_uniformity, random_deg_t_vecs
import tqdm
import logging

logger = logging.getLogger(__name__)


def transform(signal, verbosity=0, report=False, timing_verbose=False, **kwargs):
    n = signal.n
    query_method = kwargs.get("query_method")
    eps = 1e-5

    if kwargs.get("notebook"):
        pbar = tqdm.notebook.tqdm(unit=" samples")
    else:
        pbar = tqdm.tqdm(unit=" samples")

    # obtain H (subsampling matrix)
    if query_method == "simple":
        b = n
        # decoder = kwargs.get("decoder")
        decoder = lambda _, x: (x.flatten().astype(int), True)
        H = get_Ms(n, b, 2, method="simple", num_to_get=1)[0]
    elif query_method == "group_testing":
        print("Query Method is set to group testing. Ensure that wt and p are set correctly!")
        print(f"Current parameters: wt={kwargs.get('wt')}, b={kwargs.get('b')}")
        wt = kwargs.get("wt")
        t = kwargs.get("t")
        b = kwargs.get("b")
        decoder = kwargs.get("decoder")
        H = get_Ms(n, b, 2, method="group_testing", num_to_get=1, p=b, wt=wt, t=t)[0]
        mean_err, cov_err = test_uniformity(H.T, lambda x: random_deg_t_vecs(t, n, x), 50000)
        print(f"Running Uniformity Check")
        print(f"Normalized Mean L2 ={mean_err}\nNormalized Cov L2 = {cov_err}")
    else:
        raise NotImplementedError

    peeling_start = time.time()

    transform = {}

    measurements = signal.subsample(np.ones((1, n), dtype=np.int32))
    M = np.ones((1, 1), dtype=np.int32)
    loc = np.zeros((1, 0), dtype=np.int32)
    val = measurements[0] * np.ones(1)

    n_samples = 0
    sampling_time = 0

    for i in range(b):
        if len(val) == 0:
            break
        b1 = loc.shape[1]
        measurement_positions = np.array(((1 - loc) @ H[:, :b1].T + H[:, b1]) < 0.5, dtype=np.int32)
        sampling_start = time.time()
        measurements_new = signal.subsample(measurement_positions)
        sampling_time += time.time() - sampling_start
        n_samples += len(measurements_new)
        if verbosity >= 2:
            pbar.update(len(measurements_new))

        coefs_left = la.solve_triangular(M, measurements_new, lower=True)
        coefs_right = val - coefs_left
        support_first = np.where(np.abs(coefs_left) > eps)[0]
        support_second = np.where(np.abs(coefs_right) > eps)[0]
        dim1 = len(support_first)
        dim2 = len(support_second)
        M_prev = M.copy()
        M = np.zeros((dim1 + dim2, dim1 + dim2), dtype=np.int32)
        M[:dim1, :dim1] = M_prev[support_first][:, support_first]
        M[dim1:, :dim1] = M_prev[support_second][:, support_first]
        M[dim1:, dim1:] = M_prev[support_second][:, support_second]
        measurements = np.concatenate([measurements_new[support_first], measurements[support_second]])
        locs_first = np.zeros((len(support_first), b1 + 1), dtype=np.int32)
        locs_first[:, :b1] = loc[support_first]
        locs_second = np.zeros((len(support_second), b1 + 1), dtype=np.int32)
        locs_second[:, :b1] = loc[support_second]
        locs_second[:, -1] = 1
        loc = np.concatenate([locs_first, locs_second], axis=0)
        val = np.concatenate([coefs_left[support_first], coefs_right[support_second]])

    for ix, (l, m) in enumerate(zip(loc, val)):
        k_dec, success = decoder(H.T, l[np.newaxis, :].astype(bool).T)
        transform[tuple(k_dec)] = m

    peeling_time = time.time() - peeling_start
    if timing_verbose:
        logger.info(f"Peeling Time: {peeling_time}")

    loc = list(transform.keys())

    if not report:
        return transform
    else:
        if len(loc) > 0:
            loc = list(loc)
            if kwargs.get("sort", False):
                loc = sort_vecs(loc)
            avg_hamming_weight = np.mean(calc_hamming_weight(loc))
            max_hamming_weight = np.max(calc_hamming_weight(loc))
        else:
            loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
        result = {
            "transform": transform,
            "runtime": peeling_time,
            "sampling_time": sampling_time,
            "n_samples": n_samples,
            "locations": loc,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight
        }
        return result
