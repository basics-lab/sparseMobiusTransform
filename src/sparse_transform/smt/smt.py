from sparse_transform.smt.reconstruct import singleton_detection
from sparse_transform.smt.input_signal_subsampled import SubsampledSignal
from sparse_transform.smt.utils import bin_to_dec, calc_hamming_weight, dec_to_bin_vec, sort_vecs

import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


def transform(signal, reconstruct_method_source, reconstruct_method_channel, b, num_subsample, num_repeat, source_decoder,
              noise_sd=None, verbosity=0, report=False, timing_verbose=False, sort=False):
    """
     Computes the boolean Mobius transform of a signal object
     :param signal: SubsampledSignal object to be transformed.
     :param reconstruct_method_source:
     :param reconstruct_method_channel:
     :param b:
     :param num_subsample:
     :param num_repeat:
     :param noise_sd: standard deviation of the noise, used to set the bin identification cutoff.
     :param source_decoder:
     :param verbosity: Larger numbers lead to increased number of printouts
     :param report: If set to True, outputs detailed information about the amount of time each transform step takes.
     :param timing_verbose: If set to True this function returns a dictionary containing additional metrics:
      "runtime": transform_time + peeling_time, "n_samples": total number of samples,"locations": locations of nonzero indicies, "avg_hamming_weight" average
       hamming weight of non-zero indicies, "max_hamming_weight": the maximum hamming weight of a nonzero index,
       "fail_low_noise": flag indicating if the algorithm failed because of low noise setting,
       "fail_high_noise": flag indicating if the algorithm failed because of high noise setting
     :param sort: return the locations sorted in lexicographical order
     :return:
     """
    q = 2
    n = signal.n

    result = []

    mt = {}
    mt_counts = {}

    peeling_max = q ** n
    peeled = set([])
    if isinstance(signal, SubsampledSignal):
        Ms, Ds, Us, Ts = signal.get_MDU(num_subsample, num_repeat, b, trans_times=True)
    else:
        raise NotImplementedError("SMT currently only supports signals that inherit from SubsampledSignal")
    for i in range(len(Ds)):
        Us[i] = np.vstack(Us[i])
        Ds[i] = np.vstack(Ds[i])
    transform_time = np.sum(Ts)
    if timing_verbose:
        print(f"Transform Time:{transform_time}", flush=True)
    Us = np.array(Us)
    # print(Us)

    gamma = 0.5
    cutoff = 1e-9 + (1 + gamma) * (noise_sd ** 2)  # noise threshold

    # begin peeling
    # index convention for peeling: 'i' goes over all M/U/S values
    # i.e. it refers to the index of the subsampling group (zero-indexed - off by one from the paper).
    # 'j' goes over all columns of the WHT subsample matrix, going from 0 to 2 ** b - 1.
    # e.g. (i, j) = (0, 2) refers to subsampling group 0, and aliased bin 2 (10 in binary)
    # which in the example of section 3.2 is the multiton X[0110] + X[1010] + W1[10]

    # a multiton will just store the (i, j)s in a list
    # a singleton will map from the (i, j)s to the true (binary) values k.
    # e.g. the singleton (0, 0), which in the example of section 3.2 is X[0100] + W1[00]
    # would be stored as the dictionary entry (0, 0): array([0, 1, 0, 0]).
    max_iter = 1000
    iter_step = 0
    cont_peeling = True
    num_peeling = 0
    conf_fraction = 0.1

    nonzero_bins = np.ones((len(Us), len(Us[0].T)))
    bins_to_check = np.ones((len(Us), len(Us[0].T)))

    peeling_start = time.time()
    singletons = {}  # dictionary from (i, j) values to the true index of the singleton, k.

    if timing_verbose:
        start_time = time.time()
    while cont_peeling and num_peeling < peeling_max and iter_step < max_iter:
        iter_step += 1
        if verbosity >= 2:
            print('-----')
            print("iter ", iter_step, flush=True)
            # print('the measurement matrix')
            # for U in Us:
            #     print(U)
        # first step: find all the singletons and multitons.
        # multitons = []  # list of (i, j) values indicating where multitons are.
        decoders = [lambda y: source_decoder(Ds[i][1:, :], y) for i in range(len(Ds))]
        for i, (U, M, D) in enumerate(zip(Us, Ms, Ds)):
            for j in range(len(U.T)):
                cutoff_bin = len(U) * cutoff
                if bins_to_check[i, j] == 1 and nonzero_bins[i, j] == 1:
                    col = U[:, j]
                    if np.linalg.norm(col) ** 2 > cutoff_bin:
                        k, decode_success = singleton_detection(
                            col,
                            method_channel=reconstruct_method_channel,
                            method_source=reconstruct_method_source,
                            q=q,
                            source_parity=signal.get_source_parity(),
                            nso_subtype="nso1",
                            source_decoder=decoders[i]
                        )
                        if decode_success:
                            signature = 1 - ((D @ k) > 0)
                            rho = np.dot(signature, col) / sum(signature)
                            residual = col - rho * signature
                            j_qary = dec_to_bin_vec([j], b).T[0]
                            bin_matching = np.all(((M.T @ k) > 0) == j_qary)
                        else:
                            residual = float('inf')
                            rho = float('inf')
                            bin_matching = False
                        res_norm_sq = np.linalg.norm(residual) ** 2
                        if verbosity >= 5:
                            print((i, j), bin_matching, res_norm_sq, cutoff_bin)
                        if (not bin_matching) or res_norm_sq > cutoff_bin:
                            if verbosity >= 6:
                                print("We have a Multiton")
                        else:  # declare as singleton
                            singletons[(i, j)] = (k, rho, res_norm_sq)
                            if verbosity >= 3:
                                print("We have a Singleton at " + str(k))
                    else:
                        nonzero_bins[i, j] = 0
                        if verbosity >= 6:
                            print("We have a Zeroton")
                bins_to_check[i, j] = 0

        # sort and display all discovered singletons
        singletons = dict(sorted(singletons.items(), key=lambda item: item[1][2]))
        if verbosity >= 5:
            print('singletons:')
            for ston in singletons.items():
                print("\t{0} {1}\n".format(ston, bin_to_dec(ston[1][0])))

            # print("Multitons : {0}\n".format(multitons))

        # if there were no multi-tons and single-tons, terminate the algorithm
        if len(singletons) == 0:
            cont_peeling = False

        # balls to peel (only peel the best conf_fraction of singletons)
        num_to_peel = max(int(conf_fraction * len(singletons)), 1)
        singletons_to_peel = dict(list(singletons.items())[:num_to_peel])
        balls_to_peel = set()
        ball_values = {}
        for (i, j) in singletons_to_peel:
            k, rho, res = singletons_to_peel[(i, j)]
            ball = tuple(k)  # Must be a hashable type
            #qary_vec_to_dec(k, q)
            balls_to_peel.add(ball)
            ball_values[ball] = rho
            result.append((k, ball_values[ball]))

        if verbosity >= 5:
            print('these balls will be peeled')
            print(balls_to_peel)
        # peel
        for ball in balls_to_peel:
            num_peeling += 1
            k = np.array(ball)[..., np.newaxis]
            potential_peels = [(l, bin_to_dec(M.T.dot(k) > 0)[0]) for l, M in enumerate(Ms)]
            if verbosity >= 6:
                k_dec = bin_to_dec(k)
                peeled.add(int(k_dec))
                print("Processing Singleton {0} {1}".format(k_dec, k.T[0]))
                for (l, j) in potential_peels:
                    print("The singleton appears in M({0}), U({1})".format(l, j))
            for peel in potential_peels:
                signature_in_stage = 1 - ((Ds[peel[0]] @ k) > 0)
                to_subtract = ball_values[ball] * signature_in_stage.T[0]
                # print(np.linalg.norm(Us[peel[0]][:, peel[1]]), np.linalg.norm(to_subtract))
                # only peel if the bin is not a zeroton
                if nonzero_bins[peel[0], peel[1]] == 1:
                    if verbosity >= 6:
                        print("Peeled ball {0} off bin {1}".format(bin_to_dec(k), peel))
                    Us[peel[0]][:, peel[1]] -= to_subtract
                    # reactivate the bin calculations
                    bins_to_check[peel[0], peel[1]] = 1
                    if peel in singletons:
                        singletons.pop(peel)

            if verbosity >= 8:
                print("Iteration Complete: The peeled indicies are:")
                print(np.sort(list(peeled)))


    loc = set()
    for k, value in result: # iterating over (i, j)s
        loc.add(tuple(k))
        if tuple(k) in mt_counts:
            mt[tuple(k)] = (mt[tuple(k)] * mt_counts[tuple(k)] + value) / (mt_counts[tuple(k)] + 1)
            mt_counts[tuple(k)] = mt_counts[tuple(k)] + 1
        else:
            mt[tuple(k)] = value
            mt_counts[tuple(k)] = 1
    if timing_verbose:
        print(f"Peeling Time:{time.time() - start_time}", flush=True)

    peeling_time = time.time() - peeling_start

    total_bin_count = len(Us) * (q ** b)
    multiton_count = np.sum(nonzero_bins) - len(loc)
    zeroton_count = total_bin_count - len(loc) - multiton_count

    if not report:
        return mt
    else:
        n_samples = np.prod(np.shape(np.array(Us)))
        if len(loc) > 0:
            loc = list(loc)
            if sort:
                loc = sort_vecs(loc)
            avg_hamming_weight = np.mean(calc_hamming_weight(loc))
            max_hamming_weight = np.max(calc_hamming_weight(loc))
        else:
            loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
        result = {
            "transform": mt,
            "runtime": transform_time + peeling_time,
            "n_samples": n_samples,
            "locations": loc,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight,
            "fail_low_noise": len(loc) == 0 and zeroton_count <= total_bin_count // 3,
            "fail_high_noise": len(loc) == 0 and multiton_count <= total_bin_count // 3
        }
        return result
