'''
Class for computing the q-ary fourier transform of a function/signal
'''
import time
import numpy as np
from smt.utils import bin_to_dec, calc_hamming_weight, dec_to_bin_vec, sort_vecs
from scipy import linalg as la

class ASMT:
    """
    Class to encapsulate the configuration of our adaptive sparse Mobius transform algorithm.
    """
    def __init__(self, **kwargs):
        self.num_repeat = kwargs.get("num_repeat")
        self.b = kwargs.get("b")
        self.source_decoder = kwargs.get("source_decoder", None)


    def transform(self, signal, verbosity=0, report=False, timing_verbose=False, **kwargs):
        """
         Computes the sparse Mobius transform of a signal object

         Arguments
         ---------

         signal : Signal
         Signal object to be transformed.

         verbosity : int
         Larger numbers lead to increased number of printouts

         timing_verbose : Boolean
         If set to True, outputs detailed information about the amount of time each transform step takes.

         report : Boolean
         If set to True this function returns optional outputs "runtime": transform_time + peeling_time,
         "n_samples": total number of samples,"locations": locations of nonzero indicies,"avg_hamming_weight" average
          hamming weight of non-zero indicies and "max_hamming_weight": the maximum hamming weight of a nonzero index

          Returns
          -------
          gwht : dict
          Mobius transform of the input signal

          runtime : scalar
          transform time + peeling time.

          n_samples : int
          number of samples used in computing the transform.

          locations : list
          List of nonzero indicies in the transform.

          avg_hamming_weight : scalar
          Average hamming wieght of non-zero indicies.

          max_hamming_weight : int
          Max hamming weight among the non-zero indicies.
         """
        q = signal.q
        n = signal.n

        eps = 1e-5

        transform = {}
        peeling_start = time.time()

        def solve_subproblem(loc_prev, measurements_prev, M_prev):
            n1 = loc_prev.shape[1]
            measurement_positions = np.ones((loc_prev.shape[0], n), dtype=np.int32)
            measurement_positions[:, :n1] = loc_prev
            measurement_positions[:, n1] = 0
            measurements_new = signal.subsample(measurement_positions)
            rhs = np.concatenate([measurements_new[:, np.newaxis],
                                  measurements_prev[:, np.newaxis] - measurements_new[:, np.newaxis]],
                                 axis=1)

            coefs = la.solve_triangular(M_prev, rhs, lower=True)
            n_queries = len(measurements_new)
            support_first = np.where(np.abs(coefs[:, 0]) > eps)[0]
            support_second = np.where(np.abs(coefs[:, 1]) > eps)[0]
            dim1 = len(support_first)
            dim = len(support_first) + len(support_second)
            M = np.zeros((dim, dim), dtype=np.int32)
            M[:dim1, :dim1] = M_prev[support_first][:, support_first]
            M[dim1:, :dim1] = M_prev[support_second][:, support_first]
            M[dim1:, dim1:] = M_prev[support_second][:, support_second]
            measurements = np.concatenate([measurements_new[support_first], measurements_prev[support_second]])
            keys_first = measurement_positions[support_first][:, :n1 + 1]
            keys_second = measurement_positions[support_second][:, :n1 + 1]
            keys_second[:, -1] = 1
            keys = np.concatenate([keys_first, keys_second], axis=0)
            transform = np.concatenate([coefs[support_first][:, 0], coefs[support_second][:, 1]])

            return transform, keys, measurements, M, n_queries

        measurements = signal.subsample(np.ones((1, n), dtype=np.int32))
        M = np.ones((1, 1), dtype=np.int32)
        loc = np.zeros((1, 0), dtype=np.int32)
        mag = np.zeros(1)
        partition_dict = {(): measurements.copy()}

        n_samples = 0
        for i in range(n):
            if len(list(partition_dict.keys())) == 0:
                keys = np.zeros((1, n), dtype=np.int32)
                fourier_coefs = np.zeros(1, dtype=np.float64)
                break

            mag, loc, measurements, M, n_queries = solve_subproblem(loc, measurements, M)

            if verbosity >= 2:
                print('iteration %d: queries %d' % (i + 1, n_queries))
            n_samples += n_queries

        for l, m in zip(loc, mag):
            transform[tuple(l)] = m

        peeling_time = time.time() - peeling_start

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
                "n_samples": n_samples,
                "locations": loc,
                "avg_hamming_weight": avg_hamming_weight,
                "max_hamming_weight": max_hamming_weight
            }
            return result
