import numpy as np
from smt.utils import random_signal_strength_model, sort_vecs, bin_vec_to_dec, dec_to_bin_vec, imt_tensored
from smt.input_signal_subsampled import SubsampledSignal
import time


def generate_signal_mobius(n, sparsity, a_min, a_max, max_weight=None, exact_weight = False):
    """
    Generates a sparse mobius transform
    """
    max_weight = n if max_weight is None else max_weight
    N = 2 ** n
    valid_sparsity = False
    while not valid_sparsity:
        if max_weight == n:
            locq = sort_vecs(np.random.randint(2, size=(n, sparsity)).T).T
        elif not exact_weight:
            non_zero_idx_vals = np.ones((max_weight, sparsity))

            non_zero_idx_pos = np.random.choice(a=n, size=(sparsity, max_weight))
            locq = np.zeros((n, sparsity), dtype=int)
            for i in range(sparsity):
                locq[non_zero_idx_pos[i, :], i] = non_zero_idx_vals[:, i]
            locq = sort_vecs(locq.T).T
        else:
            locq = np.zeros((n, sparsity), dtype=int)
            for i in range(sparsity):
                non_zero_idx_pos = np.random.choice(a=n, size=max_weight, replace=False)
                for j in range(max_weight):
                    locq[non_zero_idx_pos[j], i] = 1
            locq = sort_vecs(locq.T).T
        loc = bin_vec_to_dec(locq)
        if len(np.unique(loc)) == sparsity:
            valid_sparsity = True
    strengths = random_signal_strength_model(sparsity, a_min, a_max)
    signal_w = dict(zip(list(map(tuple, locq.T)), strengths))
    return signal_w, locq, strengths


def get_random_subsampled_signal(n, noise_sd, sparsity, a_min, a_max, query_args, max_weight=None, noise_model=None):
    """
    Similar to get_random_signal, but instead of returning a SyntheticSignal object, it returns a SyntheticSubsampledSignal
    object. The advantage of this is that a subsampled signal does not compute the time domain signal on creation, but
    instead, creates it on the fly. This should be used (1) when n is large or (2) when sampling is expensive.
    """
    start_time = time.time()
    signal_w, loc, strengths = generate_signal_mobius(n, sparsity, a_min, a_max, max_weight=max_weight)
    signal_params = {
        "n": n,
        "noise_model": noise_model,
        "query_args": query_args
    }
    print(f"Generation Time:{time.time() - start_time}", flush=True)
    return SyntheticSubsampledSignal(signal_w=signal_w, q=2, loc=loc, strengths=strengths,
                                     noise_sd=noise_sd, **signal_params)


class SyntheticSubsampledSignal(SubsampledSignal):
    """
    This is a Subsampled signal object, except it implements the unimplemented 'subsample' function.
    """
    def __init__(self, **kwargs):

        self.n = kwargs["n"]
        self.loc = kwargs["loc"]
        self.noise_sd = kwargs["noise_sd"]
        self.noise_model = kwargs["noise_model"]
        self.strengths = kwargs["strengths"]

        def sampling_function(query_batch):
            query_indices_qary_batch = np.array(dec_to_bin_vec(query_batch, self.n)).T
            return ((((1 - query_indices_qary_batch) @ self.loc) == 0) + 0) @ self.strengths

        self.sampling_function = sampling_function

        super().__init__(**kwargs)

    def subsample(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        return self.sampling_function(query_indices)

    def subsampleShapIQ(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        base_sig = np.zeros(self.n)
        base_sig[list(query_indices)] = 1
        return self.sampling_function([bin_vec_to_dec(base_sig)])

    def get_MDU(self, ret_num_subsample, ret_num_repeat, b, trans_times=False):
        """
        wraps get_MDU method from SubsampledSignal to add synthetic noise
        """
        mdu = super().get_MDU(ret_num_subsample, ret_num_repeat, b, trans_times)
        for i in range(len(mdu[2])):
            for j in range(len(mdu[2][i])):
                size = np.array(mdu[2][i][j]).shape
                if self.noise_sd > 0:
                    if self.noise_model == "iid_spectral":
                        mdu[2][i][j] += np.random.normal(0, self.noise_sd, size=size)
                    else:
                        ValueError("Noise Model is not yet supported")
        return mdu
