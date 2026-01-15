'''
Class for computing the q-ary fourier transform of a function/signal
'''
import time
import numpy as np
from sparse_transform.smt.utils import calc_hamming_weight, sort_vecs, binary_ints, fmt
from sparse_transform.smt.query import get_Ms
import tqdm
import tqdm.notebook

from sortedcontainers import SortedDict
from collections import deque

from multiprocess import Process, Queue, Value
from queue import Empty
import logging

logger = logging.getLogger(__name__)


def get_search_loc_init(H, loc, nz, prev_depth):
    search_loc = np.ones(H.shape[1], dtype=int)
    search_loc[nz] = 0
    if prev_depth:
        H_prev_stage = H[-prev_depth:]
        loc_prev_stage = loc[-prev_depth:]
        search_loc = search_loc * np.prod(1 - H_prev_stage[np.array(loc_prev_stage) == 0, :], axis=0)
    return search_loc


def get_search_loc(H, loc, stage_depth):
    H_stage = H[-stage_depth:]
    loc_stage = loc[-stage_depth:]
    return np.all(np.array((H_stage.T + (1 - np.array(loc_stage))) % 2, dtype=bool), axis=1)


def _process_node(loc, data, H):
    nz = data["nz"]
    stage_depth = data["stage_depth"]
    prev_depth = data["prev_depth"]
    n = H.shape[1]

    if stage_depth == 0:
        search_loc = get_search_loc_init(H, loc, nz, prev_depth)
    else:
        search_loc = get_search_loc(H, loc, stage_depth)

    search_loc_count = sum(search_loc)

    if search_loc_count == 1:
        new_nz = nz + list(np.where(search_loc)[0])
        loc_new = np.zeros(n, dtype=np.int32)
        loc_new[new_nz] = 1
        output = (loc_new, True, None)
    else:
        mask = np.zeros(search_loc_count)
        # TODO choose parameters
        mask[::4] = 1
        np.random.shuffle(mask)
        h_new = np.zeros(n, dtype=np.int32)
        h_new[search_loc == 1] = mask
        output = (np.array(((1 - np.array(loc)) @ H + h_new) < 0.5, dtype=np.int32), False, h_new)

    return output


def transform_subtree(signal, peeled_func, h_mat, subtree_loc, subtree_val, return_queue, n_samples_multi, n_coefficients_multi):
    n = signal.n
    eps = 1e-10

    sampling_time = 0

    def sample_remaining_function(positions_batch):
        return signal.subsample(positions_batch) - peeled_func(positions_batch) - (((1 - positions_batch) @ loc_peeled.T) < 0.5) @ val_peeled

    loc_peeled = np.zeros((0, n), dtype=np.int32)
    val_peeled = np.zeros(0, dtype=np.int32)

    val_tree = SortedDict({tuple(subtree_loc): {"value": subtree_val, "nz": [], "stage_depth": 0, "prev_depth": 0}})
    h_tree = {tuple(subtree_loc[:i]): h_mat[:, i] for i in range(len(subtree_loc))}

    n_samples = 0
    transform_dict = {}

    while len(val_tree) > 0:
        loc, data = val_tree.popitem(index=0)
        val = data["value"]
        if abs(val) < eps:
            continue
        nz = data["nz"]
        sd = data["stage_depth"]
        pd = data["prev_depth"]
        H = np.stack([h_tree[tuple(loc[:i])] for i in range(len(loc))], axis=0)
        measurement_loc, to_peel, h_new = _process_node(loc, data, H)
        h_tree[loc] = h_new
        measurement_start = time.time()
        measurement_new = sample_remaining_function(measurement_loc[np.newaxis, :])[0]
        n_samples += 1
        n_samples_multi.value = n_samples_multi.value + 1
        sampling_time += time.time() - measurement_start
        if to_peel:
            # peel and update the node
            new_nz = list(np.where(measurement_loc)[0])
            if np.abs(measurement_new) > eps:
                loc_peeled = np.concatenate([loc_peeled, measurement_loc[np.newaxis, :]], axis=0)
                val_peeled = np.append(val_peeled, [measurement_new])
                transform_dict[tuple(measurement_loc)] = measurement_new
                n_coefficients_multi.value = n_coefficients_multi.value + 1
                new_data = {"value": data["value"] - measurement_new, "nz": new_nz, "stage_depth": 0, "prev_depth": sd}
            else:
                new_data = {"value": data["value"], "nz": new_nz, "stage_depth": 0, "prev_depth": sd}
            val_tree[loc] = new_data
        else:
            # push right child
            if abs(val - measurement_new) > eps:
                new_data = {"value": val - measurement_new, "nz": nz, "stage_depth": sd + 1, "prev_depth": pd}
                val_tree[loc + (1,)] = new_data

            # push left child
            if abs(measurement_new) > eps:
                new_data = {"value": measurement_new, "nz": nz, "stage_depth": sd + 1, "prev_depth": pd}
                val_tree[loc + (0,)] = new_data

    result = {
        "transform": transform_dict,
        "n_samples": n_samples
    }

    return_queue.put((subtree_loc, result))


def transform(signal, verbosity=0, report=False, timing_verbose=False, **kwargs):
    transformer = FASMTParallel()
    return transformer.transform(signal, verbosity=verbosity, report=report, timing_verbose=timing_verbose, **kwargs)


class FASMTParallel:

    def __init__(self):
        self.t = None
        self.sampling_time = None
        self.forest_subset_rem = None
        self.n_batches = None
        self.na_forest = None
        self.na_degree = None
        self.n_samples = None
        self.val_peeled = None
        self.loc_peeled = None
        self.pbar = None
        self.n = None
        self.b = None
        self.signal = None
        self.eps = 1e-10
        self.transform_dict = {}
        self.h_mat = None
        self.input_queue = None
        self.return_queue = None

    def transform(self, signal, verbosity, report, timing_verbose, **kwargs):
        self.n = signal.n
        self.b = kwargs.get("b", 1)
        self.t = kwargs.get("t", 10)
        self.signal = signal
        self.n_samples = 0
        self.n_batches = 0

        if kwargs.get("notebook"):
            self.pbar = tqdm.notebook.tqdm(unit=" samples")
        else:
            self.pbar = tqdm.tqdm(unit=" samples")

        query_method = kwargs.get("query_method")
        if query_method == "group_testing":
            pass
        else:
            raise NotImplementedError

        self.loc_peeled = np.zeros((0, self.n), dtype=np.int32)
        self.val_peeled = np.zeros(0, dtype=np.int32)
        self.na_forest = SortedDict()

        self.sampling_time = 0

        peeling_start = time.time()

        self.construct_forest()

        self.forest_subset_rem = {loc_subtree: sum(loc_subtree) for loc_subtree in self.na_forest.keys()}
        self.input_queue = deque()
        self.input_queue.append((0,) * self.b)
        num_processes = 16
        running_processes = 0

        return_queue = Queue()
        n_samples_multi = Value('i', self.n_samples)
        n_coefficients_multi = Value('i', 0)

        while len(self.na_forest) > 0:

            while running_processes < num_processes and self.input_queue:

                loc_subtree = self.input_queue.pop()

                def peeled_function(positions_batch):
                    return (((1 - positions_batch) @ self.loc_peeled.T) < 0.5) @ self.val_peeled

                p = Process(target=transform_subtree, args=(signal, peeled_function, self.h_mat, loc_subtree,
                                                            self.na_forest[loc_subtree], return_queue, n_samples_multi, n_coefficients_multi))
                p.start()
                running_processes += 1
                del self.na_forest[loc_subtree]

            while True:
                try:
                    loc_subtree, result = return_queue.get_nowait()
                    running_processes -= 1
                    subtree_transform = result["transform"]
                    self.n_samples += result["n_samples"]
                    self.n_batches += result["n_samples"]
                    if len(subtree_transform) > 0:
                        self.transform_dict.update(subtree_transform)
                        self.loc_peeled = np.concatenate([self.loc_peeled, np.array(list(subtree_transform.keys()))], axis=0, dtype=np.int32)
                        self.val_peeled = np.concatenate([self.val_peeled, np.array(list(subtree_transform.values())).flatten()], axis=0)

                    # reduce forest_subset_rem for all direct supersets
                    for j in range(self.b):
                        if loc_subtree[j] == 0:
                            loc_subtree_ss = list(loc_subtree)
                            loc_subtree_ss[j] = 1
                            self.forest_subset_rem[tuple(loc_subtree_ss)] -= 1
                            if self.forest_subset_rem[tuple(loc_subtree_ss)] == 0:
                                self.input_queue.append(tuple(loc_subtree_ss))
                                
                except Empty:
                    break

            self.pbar.update(n_samples_multi.value - self.pbar.n)
            self.pbar.set_postfix({"Coefficients Found": n_coefficients_multi.value})
            # wait 1 ms before next loop
            time.sleep(0.001)

        peeling_time = time.time() - peeling_start
        if timing_verbose:
            logger.info(f"Peeling Time: {peeling_time}")

        loc = list(self.transform_dict.keys())

        if not report:
            return self.transform_dict
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
                "transform": self.transform_dict,
                "runtime": peeling_time,
                "sampling_time": np.nan,
                "n_samples": self.n_samples,
                "n_batches": self.n_batches,
                "locations": loc,
                "avg_hamming_weight": avg_hamming_weight,
                "max_hamming_weight": max_hamming_weight
            }
            return result

    def subsample(self, measurement_position):
        sampling_start = time.time()
        measurement_new = self.signal.subsample(measurement_position)
        self.sampling_time += time.time() - sampling_start
        self.n_samples += len(measurement_position)
        self.n_batches += 1
        self.pbar.update(len(measurement_position))
        return measurement_new

    def construct_forest(self):
        # TODO choose parameters
        H = get_Ms(self.n, self.b, 2, method="group_testing", num_to_get=1, p=10 * self.b, wt=np.log(2), t=self.t)[0]
        L = np.array(binary_ints(self.b))
        measurement_position = (1 - ((H @ (1 - L)) > 0)).T
        samples = self.subsample(measurement_position)
        coefficients = fmt(samples)
        self.h_mat = H

        for i in range(2 ** self.b):
            val = coefficients[i]
            loc = tuple(L[:, i])
            self.na_forest[loc] = val