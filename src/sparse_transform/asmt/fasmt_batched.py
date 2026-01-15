'''
Class for computing the q-ary fourier transform of a function/signal
'''
import time
import numpy as np
from sparse_transform.smt.utils import calc_hamming_weight, sort_vecs, binary_ints, fmt
from sparse_transform.smt.query import get_Ms
import tqdm
import tqdm.notebook
from multiprocessing import Pool
from itertools import starmap
from prefetch_generator import BackgroundGenerator

from sortedcontainers import SortedDict
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


def transform(signal, verbosity=0, report=False, timing_verbose=False, **kwargs):
    transformer = FASMTBatched()
    return transformer.transform(signal, verbosity=verbosity, report=report, timing_verbose=timing_verbose, **kwargs)


class FASMTBatched:

    def __init__(self, parallel=False):
        self.t = None
        self.parallel = parallel
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
        self.eps = 1e-5
        self.transform_dict = {}
        self.h_mat = None
        if self.parallel:
            self.pool = Pool()

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
        self.h_mat = SubsampleMatrix(self.n)

        self.sampling_time = 0

        peeling_start = time.time()

        self.construct_forest()
        self.forest_subset_rem = {loc_subtree: sum(loc_subtree) for loc_subtree in self.na_forest.keys()}

        while len(self.na_forest) > 0:

            nodes_to_process = []
            subtree_list = list(self.na_forest.keys())

            # delete zero nodes and empty subtrees
            for loc_subtree in subtree_list:
                if self.forest_subset_rem[loc_subtree] == 0:
                    subtree = self.na_forest[loc_subtree]
                    while len(subtree) > 0:
                        loc, data = subtree.peekitem(0)
                        # delete zero nodes
                        if abs(data["value"]) < self.eps:
                            del subtree[loc]
                        else:
                            break
                    # delete empty subtrees
                    if len(subtree) == 0:
                        del self.na_forest[loc_subtree]
                        # reduce forest_subset_rem for all direct supersets
                        for j in range(self.b):
                            if loc_subtree[j] == 0:
                                loc_subtree_ss = list(loc_subtree)
                                loc_subtree_ss[j] = 1
                                self.forest_subset_rem[tuple(loc_subtree_ss)] -= 1

            # obtain nodes that can be processed
            for loc_subtree in self.na_forest.keys():
                # check if subsets are complete
                if self.forest_subset_rem[loc_subtree] == 0:
                    subtree = self.na_forest[loc_subtree]
                    loc, data = subtree.peekitem(0)
                    nodes_to_process.append((loc, data))

            if len(nodes_to_process) == 0:
                continue

            # obtain sample locations for nodes to be processed
            pos_list, peel_list, h_row_list = self._process_node_parallel(nodes_to_process)
            measurement_position = np.vstack(pos_list)

            measurement_new = self.subsample(measurement_position)

            # update the tree
            for i, (loc, data) in enumerate(nodes_to_process):
                # pop leftmost node
                del self.na_forest[loc[:self.b]][loc]
                val = data["value"]
                nz = data["nz"]
                sd = data["stage_depth"]
                pd = data["prev_depth"]
                h_rows = data["h_rows"]
                if peel_list[i]:
                    # peel and update the node
                    new_nz = list(np.where(pos_list[i])[0])
                    if np.abs(measurement_new[i]) > self.eps:
                        self.loc_peeled = np.concatenate([self.loc_peeled, pos_list[i][np.newaxis, :]], axis=0)
                        self.val_peeled = np.append(self.val_peeled, [measurement_new[i]])
                        self.transform_dict[tuple(pos_list[i])] = measurement_new[i]
                        new_data = {"value": val - measurement_new[i], "nz": new_nz, "stage_depth": 0, "prev_depth": sd, "h_rows": h_rows}
                    else:
                        new_data = {"value": val, "nz": new_nz, "stage_depth": 0, "prev_depth": sd, "h_rows": h_rows}
                    self.push_node(loc, new_data)
                else:
                    # push right child
                    if abs(val - measurement_new[i]) > self.eps:
                        new_data = {"value": val - measurement_new[i], "nz": nz, "stage_depth": sd + 1, "prev_depth": pd, "h_rows": h_rows + [h_row_list[i]]}
                        self.push_node(loc + (1,), new_data)

                    # push left child
                    if abs(measurement_new[i]) > self.eps:
                        new_data = {"value": measurement_new[i], "nz": nz, "stage_depth": sd + 1, "prev_depth": pd, "h_rows": h_rows + [h_row_list[i]]}
                        self.push_node(loc + (0,), new_data)

            self.pbar.set_postfix({"Batch Size": len(nodes_to_process), "Coefficients Found": len(self.val_peeled)})

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
                "sampling_time": self.sampling_time,
                "n_samples": self.n_samples,
                "n_batches": self.n_batches,
                "locations": loc,
                "avg_hamming_weight": avg_hamming_weight,
                "max_hamming_weight": max_hamming_weight
            }
            return result

    def subsample(self, measurement_position):
        sampling_start = time.time()
        measurement_new = self.signal.subsample(measurement_position) - self.sample_peeled_function(measurement_position)
        self.sampling_time += time.time() - sampling_start
        self.n_samples += len(measurement_position)
        self.n_batches += 1
        self.pbar.update(len(measurement_position))
        return measurement_new

    def sample_peeled_function(self, positions_batch):
        return (((1 - positions_batch) @ self.loc_peeled.T) < 0.5) @ self.val_peeled

    def construct_forest(self):
        # TODO choose parameters
        H = get_Ms(self.n, self.b, 2, method="group_testing", num_to_get=1, p=10*self.b, wt=np.log(2), t=self.t)[0]
        L = np.array(binary_ints(self.b))
        measurement_position = (1 - ((H @ (1 - L)) > 0)).T
        samples = self.subsample(measurement_position)
        coefficients = fmt(samples)
        row_nums = []
        for j in range(self.b):
            rn = self.h_mat.add_row(H[:, j])
            row_nums.append(rn)

        for i in range(2 ** self.b):
            val = coefficients[i]
            loc = tuple(L[:, i])
            self.na_forest[loc] = SortedDict({loc: {"value": val, "nz": [], "stage_depth": 0, "prev_depth": None, "h_rows": row_nums}})

    def _process_node_parallel(self, nodes_to_process):
        pos_list = []
        peel_list = []
        h_row_list = []
        inputs = ((loc, data, self.h_mat.array[data["h_rows"]]) for loc, data in nodes_to_process)
        if len(nodes_to_process) > 5:
            inputs = BackgroundGenerator(inputs, max_prefetch=3)
        process_iterator = self.pool.starmap if self.parallel else starmap
        for measurement_loc, to_peel, h_new in process_iterator(_process_node, inputs):
            if h_new is not None:
                rn = self.h_mat.add_row(h_new)
                h_row_list.append(rn)
            else:
                h_row_list.append(None)
            pos_list.append(measurement_loc)
            peel_list.append(to_peel)
        return pos_list, peel_list, h_row_list

    def push_node(self, loc, data):
        self.na_forest[loc[:self.b]][loc] = data

    def peel_node(self, loc, data, loc_new, measurement_peel):
        new_nz = list(np.where(loc_new)[0])
        if np.abs(measurement_peel) > self.eps:
            self.loc_peeled = np.concatenate([self.loc_peeled, loc_new[np.newaxis, :]], axis=0)
            self.val_peeled = np.append(self.val_peeled, [measurement_peel])
            self.transform_dict[tuple(loc_new)] = measurement_peel
            new_data = {"value": data["value"] - measurement_peel, "nz": new_nz, "stage_depth": 0}
        else:
            new_data = {"value": data["value"], "nz": new_nz, "stage_depth": 0}
        self.na_forest[loc[:self.b]][loc] = new_data


class SubsampleMatrix:

    def __init__(self, n):
        self.n = n
        self.array = np.zeros((1000, n), dtype=int)
        self.size = 1000
        self.used = 0

    def add_row(self, row):
        if self.used == self.size:
            array_new = np.zeros((2 * self.size, self.n), dtype=int)
            array_new[:self.size] = self.array
            self.size = 2 * self.size
            self.array = array_new
        row_number = self.used
        self.array[self.used] = row
        self.used += 1
        return row_number
