'''
Class for computing the q-ary fourier transform of a function/signal
'''
import time
import numpy as np
from sparse_transform.smt.utils import calc_hamming_weight, sort_vecs
import tqdm
import tqdm.notebook

from sortedcontainers import SortedDict


def transform(signal, verbosity=0, report=False, timing_verbose=False, **kwargs):
    n = signal.n
    query_method = kwargs.get("query_method")
    eps = 1e-5

    if query_method == "group_testing":
        pass
    else:
        raise NotImplementedError

    peeling_start = time.time()
    sampling_time = 0

    transform = {}

    def sample_peeled_function(positions_batch):
        return (((1 - positions_batch) @ loc_peeled.T) < 0.5) @ val_peeled

    loc_peeled = np.zeros((0, n), dtype=np.int32)
    val_peeled = np.zeros(0, dtype=np.int32)

    val_tree = SortedDict({tuple(): {"value": signal.subsample(np.ones((1, n))), "nz": [], "stage_depth": 0}})
    h_tree = {}

    n_samples = 0
    if kwargs.get("notebook"):
        pbar = tqdm.notebook.tqdm(unit=" samples")
    else:
        pbar = tqdm.tqdm(unit=" samples")

    while len(val_tree) > 0:

        loc, data = val_tree.popitem(index=0)
        val = data["value"]
        if val == 0:
            continue
        nz = data["nz"]
        stage_depth = data["stage_depth"]
        h_new = np.zeros(n)
        if len(loc) > 0:
            H = np.stack([h_tree[tuple(loc[:i])] for i in range(len(loc))], axis=0)
            if stage_depth == 0:
                search_loc = [i for i in range(n) if i not in nz]
            else:
                H_stage = H[-stage_depth:]
                loc_stage = loc[-stage_depth:]
                search_loc = np.where(np.prod((H_stage.T + (1 - np.array(loc_stage))) % 2, axis=1))[0]
            h_new = np.zeros(n)
            mask_loc = np.random.choice(search_loc, size=(len(search_loc) + 1) // 2, replace=False)
            h_new[mask_loc] = 1
            measurement_position = np.array(((1 - np.array(loc)) @ H + h_new) < 0.5, dtype=np.int32)
        else:  # root node only
            search_loc = np.arange(n)
            mask_loc = np.random.choice(search_loc, size=len(search_loc) // 2, replace=False)
            h_new[mask_loc] = 1
            measurement_position = np.array(h_new < 0.5, dtype=np.int32)
        h_tree[loc] = h_new
        sampling_start = time.time()
        measurement_new = signal.subsample(measurement_position[np.newaxis, :]) - sample_peeled_function(measurement_position[np.newaxis, :])
        n_samples += 1
        pbar.update()
        sampling_time += time.time() - sampling_start

        if abs(measurement_new) > eps:
            val_tree[loc + (0,)] = {"value": measurement_new, "nz": nz, "stage_depth": stage_depth + 1}
        if abs(val - measurement_new) > eps:
            if len(search_loc) == 1:
                new_nz = nz + [search_loc[0]]
                loc_new = np.zeros(n, dtype=np.int32)
                loc_new[new_nz] = 1
                measurement_new = signal.subsample(loc_new[np.newaxis, :]) - sample_peeled_function(loc_new[np.newaxis, :])
                n_samples += 1
                pbar.update()
                if np.abs(measurement_new) > eps:
                    loc_peeled = np.concatenate([loc_peeled, loc_new[np.newaxis, :]], axis=0)
                    val_peeled = np.append(val_peeled, [measurement_new])
                    pbar.set_postfix({"Coefficients Found": len(val_peeled)})
                    transform[tuple(loc_new)] = measurement_new

                val_tree[loc + (1,)] = {"value": val - measurement_new, "nz": new_nz, "stage_depth": 0}
            else:
                val_tree[loc + (1,)] = {"value": val - measurement_new, "nz": nz, "stage_depth": stage_depth + 1}

        pass

    peeling_time = time.time() - peeling_start
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
            "n_batches": n_samples,
            "locations": loc,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight
        }
        return result
