# import algorithms
from sparse_transform import smt, asmt, fasmt, fasmt_batched, fasmt_parallel
from sparse_transform.noise_estimator import NoiseEstimator

# import signal objects
from sparse_transform.smt.input_signal_subsampled import SubsampledSignal as MobiusSubsampledSignal
from sparse_transform.smt.input_signal import Signal as MobiusSignal

# import decoders
from sparse_transform.smt.random_group_testing import decode, decode_robust

import logging

logger = logging.getLogger(__name__)


def run_transforms(func, algo, **kwargs):
    if algo == "SMT":
        config = _parse_smt_parameters(**kwargs)
    elif algo == "ASMT":
        config = _parse_asmt_parameters(**kwargs)
    elif algo == "FASMT":
        config = {"algo": fasmt.transform, "transform_args": {"query_method": "group_testing", "decoder": decode}, "name": kwargs.get('name', "FASMT")}
    elif algo == "FASMTBatched":
        config = {"algo": fasmt_batched.transform, "transform_args": {"query_method": "group_testing", "decoder": decode, "b": kwargs['b'],
                                                                      "t": kwargs["t"]}, "name": kwargs.get('name', "FASMTBatched")}
    elif algo == "FASMTParallel":
        config = {"algo": fasmt_parallel.transform, "transform_args": {"query_method": "group_testing", "decoder": decode, "b": kwargs['b'],
                                                                       "t": kwargs["t"]}, "name": kwargs.get('name', "FASMTParallel")}
    else:
        raise NotImplementedError

    logger.info(f'Running Tests with {config}')

    if kwargs.get("noise_sd") is None and algo in ["SMT"]:
        logger.warning(f"noise_sd is set to None, the algorithm will search for the best value.")
        estimate_noise = True
    else:
        logger.info(f"noise_sd is set to {kwargs.get('noise_sd')}.")
        estimate_noise = False

    result = _run_algorithm(func, config, kwargs['n'], kwargs.get('q', 2), estimate_noise, kwargs.get('noise_interval'), kwargs.get('noise_step'))

    return result


def _run_algorithm(func, config, n, q, estimate_noise, noise_interval, noise_step):
    algo = config["algo"]
    signal_generator = _get_signal_generator(algo)

    if "query_args" in config:  # obtain training samples
        signal = signal_generator(func=func, n=n, q=q, query_args=config["query_args"])
    else:  # create a signal object without sampling
        signal = signal_generator(func=func, n=n, q=q)

    # estimate the noise level
    if estimate_noise:
        test_query_args = config["query_args"]
        test_query_args["subsampling_method"] = "uniform"
        test_query_args["n_samples"] = 1000
        test_signal = signal_generator(func=func, n=n, q=q, query_args=test_query_args)

        noise_helper = NoiseEstimator(algo, n=n, q=q,
                                      transform_args=config["transform_args"],
                                      query_args=config["query_args"],
                                      train_signal=signal,
                                      test_signal=test_signal,
                                      noise_interval=noise_interval,
                                      noise_step=noise_step)
        peeling_noise = noise_helper.estimate_noise()
        config["transform_args"]["noise_sd"] = peeling_noise
    return algo(signal, verbosity=0, timing_verbose=False, report=True, sort=True, **config["transform_args"])


def _get_signal_generator(algo):
    if algo == smt.transform:
        return MobiusSubsampledSignal
    elif algo in [asmt.transform, fasmt.transform, fasmt_batched.transform, fasmt_parallel.transform]:
        return MobiusSignal
    else:
        raise NotImplementedError

def _parse_smt_parameters(**kwargs):
    if 't' in kwargs:  # coded SMT
        query_method = "group_testing"
        if kwargs.get('noise_sd') == 0:  # noiseless coded SMT
            delays_method_source = "coded"
            delays_method_channel = "identity"
            source_decoder = decode
        else:  # noisy coded SMT
            delays_method_source = "coded"
            delays_method_channel = "nso"
            source_decoder = _robust_source_decoder
    else:  # non-coded SMT
        query_method = "simple"
        if kwargs.get('noise_sd') == 0:  # noiseless non-coded SMT
            delays_method_source = "identity"
            delays_method_channel = "identity"
            source_decoder = None
        else:  # TODO: implement noisy non-coded SMT
            raise NotImplementedError

    query_args = {
        "query_method": query_method,
        "num_subsample": kwargs['num_subsample'],
        "delays_method_source": delays_method_source,
        "subsampling_method": "smt",
        "delays_method_channel": delays_method_channel,
        "num_repeat": kwargs['num_repeat'],
        "b": kwargs['b']
    }

    transform_args = {
        "num_subsample": kwargs['num_subsample'],
        "num_repeat": kwargs['num_repeat'],
        "reconstruct_method_source": delays_method_source,
        "reconstruct_method_channel": delays_method_channel,
        "b": kwargs['b'],
        "noise_sd": kwargs.get('noise_sd', None),
    }

    if 't' in kwargs:
        query_args['t'] = kwargs['t']
        query_args['wt'] = kwargs.get('wt', 0.9)
        query_args['p'] = kwargs.get('p', 100)
        transform_args["source_decoder"] = source_decoder

    config = {"algo": smt.transform, "name": kwargs.get('name', "SMT"), "query_args": query_args, "transform_args": transform_args}
    return config


def _parse_asmt_parameters(**kwargs):
    if kwargs.get('b'):
        transform_args = {"query_method": "group_testing", "b": kwargs['b'], "wt": 0.9, "t": kwargs['t'], "decoder": decode}
    else:
        transform_args = {"query_method": "simple"}
    return {"algo": asmt.transform, "transform_args": transform_args, "name": kwargs.get('name', "ASMT")}


def _robust_source_decoder(D, y):
    dec, err, decode_success = decode_robust(D, y, 1, solution=None)
    return dec, decode_success
