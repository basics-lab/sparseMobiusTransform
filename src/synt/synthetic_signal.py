import numpy as np
from sparse_transform.smt.utils import random_signal_strength_model, sort_vecs, bin_vec_to_dec, dec_to_bin_vec, imt_tensored
from sparse_transform.smt.input_signal_subsampled import SubsampledSignal
from sparse_transform.smt.input_signal import Signal
import time


def generate_signal_mobius(n, sparsity, a_min, a_max, max_weight=None, exact_weight=False):
    """
    Generates a sparse mobius transform
    """
    max_weight = n if max_weight is None else max_weight
    unique_locq = set()
    for i in range(sparsity):
        if max_weight == n:
            new_loc = np.random.randint(2, size=n)
        elif not exact_weight:
            new_loc = np.zeros(n, dtype=np.int32)
            weight = np.random.choice(a=max_weight) + 1
            non_zero_idx = np.random.choice(a=n, size=weight)
            new_loc[non_zero_idx] = 1
        else:
            new_loc = np.zeros(n, dtype=np.int32)
            non_zero_idx = np.random.choice(a=n, size=max_weight)
            new_loc[non_zero_idx] = 1
        unique_locq.add(tuple(new_loc))
    locq = np.array(list(unique_locq), dtype=np.int32).T
    locq = sort_vecs(locq.T).T
    strengths = random_signal_strength_model(locq.shape[1], a_min, a_max)
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


def get_random_signal(n, noise_sd, sparsity, a_min, a_max, max_weight=None, noise_model=None):
    """
    Similar to get_random_signal, but instead of returning a SyntheticSignal object, it returns a SyntheticSubsampledSignal
    object. The advantage of this is that a subsampled signal does not compute the time domain signal on creation, but
    instead, creates it on the fly. This should be used (1) when n is large or (2) when sampling is expensive.
    """
    start_time = time.time()
    signal_w, loc, strengths = generate_signal_mobius(n, sparsity, a_min, a_max, max_weight=max_weight)
    print(f"Generation Time:{time.time() - start_time}", flush=True)
    return SyntheticSignal(n=n, noise_model=noise_model, signal_w=signal_w, q=2, loc=loc,
                           strengths=strengths, noise_sd=noise_sd)


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


class SyntheticSignal(Signal):
    """
    This is a signal object, except it implements the unimplemented 'subsample' function.
    """

    def __init__(self, **kwargs):

        self.n = kwargs["n"]
        self.loc = kwargs["loc"]
        self.noise_sd = kwargs["noise_sd"]
        self.noise_model = kwargs["noise_model"]
        self.strengths = kwargs["strengths"]

        def sampling_function(query_batch):
            return ((((1 - query_batch) @ self.loc) == 0) + 0) @ self.strengths

        self.sampling_function = sampling_function

        super().__init__(**kwargs)

    def subsample(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        return self.sampling_function(query_indices)
