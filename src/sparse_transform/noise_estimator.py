'''
Class that perfroms noise search by running repeated tests
'''
import numpy as np
import logging
from sparse_transform.smt.input_signal_subsampled import SubsampledSignal as MobiusSubsampledSignal

logger = logging.getLogger(__name__)


class NoiseEstimator:
    '''
    Class to estimate peeling noise

    Attributes
    ---------
    n, q: int, they define the problem
    sft: QSFT object that we can use to peel 
    transform_args: dictionary required to run tests
    query_args: distionary required to run tests
    func: the oracle, required to run tests
    signal: the signal object, required to run tests. 
    '''

    def __init__(self, algo, n, q, transform_args, query_args, train_signal, test_signal, noise_interval=(-3, 3), noise_step=0.2) -> None:
        # signal parameters:
        self.n = n
        self.q = q
        self.algo = algo
        self.transform_args = transform_args
        self.query_args = query_args
        self.train_signal = train_signal
        self.test_signal = test_signal

        # algorithm parameters:
        # max number of iterations to find the search interval (first phase)
        self.iter_1 = 100
        # max number of iterations to find the best value within the search interval (second phase)
        self.iter_2 = 20
        # counter is in relation to iter_2
        self.counter_2 = 0
        # log10 step size during the first phase
        self.step = noise_step
        self.noise_interval = noise_interval
        # minimal distance between left and right before we call it a stop
        self.min_dist = 0.001
        # ratio is golden ratio
        self.ratio = (np.sqrt(5) + 1) / 2
        # This is the threshold of RMSE I want. If at some point I see that RMSE is less than this number,I'm done.
        self.threshold = 1e-5
        # cache for test results
        self.cache = {}

    def estimate_noise(self):
        """
        Find the best noise level for a given problem
        :return: float32
        """

        logger.info(f"Searching for the best value for noise_sd")

        left, right = self.find_gap()

        logger.info(f"Phase 1 resulted in noise interval: [{10 ** left}, {10 ** right}]")

        if left == right:
            logger.info(f"Phase 2 resulted in noise: {10 ** left}")
            return 10 ** left
        else:
            try:
                answer = self.find_noise(left, right)
                logger.info(f"Phase 2 resulted in noise: {10 ** answer}")
                return 10 ** answer
            except TimeoutError:
                return None

    # a function that simplifies running the test
    def _test_noise(self, log_noise):
        log_noise = round(log_noise, 3)
        if log_noise not in self.cache:
            self.transform_args["noise_sd"] = 10 ** log_noise
            result = self.algo(self.train_signal, verbosity=0, timing_verbose=False, report=True, sort=True, **self.transform_args)
            error, fail_low, fail_high = self._estimate_error(result["transform"]), result["fail_low_noise"], result["fail_high_noise"]
            self.cache[log_noise] = (error, fail_low, fail_high)
        return self.cache[log_noise]

    # a function that estimates the RMSE
    def _estimate_error(self, transform):

        new_signal = self.test_signal.signal_t
        (sample_idx, samples) = list(new_signal.keys()), list(new_signal.values())
        batch_size = 10000

        if len(transform.keys()) == 0:
            return np.linalg.norm(np.array(samples)) / np.sqrt(len(samples))

        y_hat = []
        for i in range(0, len(sample_idx), batch_size):
            sample_idx_batch = sample_idx[i:i + batch_size]
            if isinstance(self.test_signal, MobiusSubsampledSignal):
                y_hat.append(self._sample_mobius_function(transform, sample_idx_batch))
            else:
                raise ValueError

        y_hat = np.concatenate(y_hat)

        return np.linalg.norm(np.array(y_hat) - np.array(samples)) / np.sqrt(len(samples))

    # This function finds the interval that the true noise lies in.
    def find_gap(self):

        log_init = (self.noise_interval[0] + self.noise_interval[1]) / 2.0
        log_noise = log_init
        results = {}

        while True:
            # Run the test
            error, fail_low, fail_high = self._test_noise(log_noise)
            logger.info(f"Phase 1 -- noise level: {10 ** log_noise}, RMSE: {error}, fail_low: {fail_low}, fail_high: {fail_high}")
            results[log_noise] = error
            if fail_high or log_noise > self.noise_interval[1]:
                break
            log_noise = log_noise + self.step

        log_noise = log_init - self.step
        while True:
            error, fail_low, fail_high = self._test_noise(log_noise)
            logger.info(f"Phase 1 -- noise level: {10 ** log_noise}, RMSE: {error}, fail_low: {fail_low}, fail_high: {fail_high}")
            results[log_noise] = error
            if fail_low or log_noise < self.noise_interval[0]:
                break
            log_noise = log_noise - self.step

        p1 = sorted(list(results.keys()), key=lambda x: results[x])[0]
        return p1 - self.step, p1 + self.step

    # Given an interval, this function helps finding the true noise. We do a golden search here.
    def find_noise(self, left, right) -> float:

        assert left < right

        self.counter_2 = self.counter_2 + 1
        if self.counter_2 >= self.iter_2:
            raise TimeoutError

        # Calculate the middles which is the goldren ratio whatever
        mid_1 = right - (right - left) / self.ratio
        mid_2 = left + (right - left) / self.ratio
        assert mid_1 < mid_2

        # evaluate on all four points
        test_points = [left, mid_1, mid_2, right]
        errors = []
        fail_left = []
        fail_right = []
        for tp in test_points:
            e, fl, fr = self._test_noise(tp)
            errors.append(e)
            fail_left.append(fl)
            fail_right.append(fr)

        # Determine if at any of these noises peeling is successful
        for i in range(3):
            if (errors[i]) <= self.threshold:
                self.answer_error = errors[i]
                return test_points[i]

        logger.info(f"Phase 2 Iteration {self.counter_2} -- noise interval: [{(10 ** test_points[1]):4f}, {(10 ** test_points[2]):4f}] "
                    f"-- RMSE: [{(errors[1]):4f}, {(errors[2]):4f}]")

        # Terminate if left is too close to right
        if right - left < self.min_dist:
            self.answer = left
            self.answer_error = errors[0]
            return left

        # if it failed at middle points
        if fail_left[1]:
            return self.find_noise(mid_1, right)
        elif fail_right[1]:
            return self.find_noise(left, mid_1)
        elif fail_left[2]:
            return self.find_noise(mid_2, right)
        elif fail_right[2]:
            return self.find_noise(left, mid_2)
        else:  # no issues -- just compare mid points
            if errors[1] < errors[2]:
                return self.find_noise(left, mid_2)
            else:
                return self.find_noise(mid_1, right)

    def _sample_mobius_function(self, transform, sample_idx_batch):
        beta_keys = list(transform.keys())
        beta_values = list(transform.values())
        return ((1 - np.array(sample_idx_batch)) @ np.array(beta_keys).T < 0.5) @ np.array(beta_values)
    
