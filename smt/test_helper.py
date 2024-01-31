import numpy as np
from smt.lasso import lasso_decode
from smt.qsft import QSFT
from smt.utils import NpEncoder, dec_to_bin_vec, bin_vec_to_dec
import json
from smt.random_group_testing import decode_robust


class TestHelper:

    def __init__(self, signal_args, methods, query_args, test_args, exp_dir, subsampling=True):

        self.n = signal_args["n"]
        self.q = signal_args["q"]

        self.exp_dir = exp_dir
        self.subsampling = subsampling

        config_path = self.exp_dir / "config.json"
        config_exists = config_path.is_file()

        if not config_exists:
            config_dict = {"query_args": query_args}
            with open(config_path, "w") as f:
                json.dump(config_dict, f, cls=NpEncoder)

        self.signal_args = signal_args
        self.query_args = query_args
        self.test_args = test_args

        if self.subsampling:
            if "smt" in methods:
                self.train_signal = self.load_train_data()
            elif "smt_coded" in methods:
                self.train_signal = self.load_train_data_coded(robust=False)
            elif "smt_robust" in methods:
                self.train_signal = self.load_train_data_coded(robust=True)
            self.test_signal = self.load_test_data()
            # print("Test data loaded.", flush=True)
        else:
            self.train_signal = self.load_full_data()
            self.test_signal = self.train_signal
            if any([m.startswith("binary") for m in methods]):
                raise NotImplementedError  # TODO: implement the conversion
            # print("Full data loaded.", flush=True)

    def generate_signal(self, signal_args):
        raise NotImplementedError

    def load_train_data(self):
        self.query_args.query_args.update({
            "subsampling_method": "smt",
            "query_method": "simple",
            "delays_method_source": "identity",
            "delays_method_channel": "identity"
        })
        self.signal_args["folder"] = self.exp_dir / "train"
        self.signal_args["query_args"] = self.query_args

        return self.generate_signal(self.signal_args)

    def load_train_data_coded(self, robust):
        if robust:
            self.query_args.update({
                "query_method": "group_testing",
                "delays_method_source": "coded",
                "subsampling_method": "smt",
                "delays_method_channel": "nso",
                "t": self.signal_args["t"]
            })
        else:
            self.query_args.update({
                "subsampling_method": "smt",
                "query_method": "complex",
                "delays_method_source": "coded",
                "delays_method_channel": "nso",
                "t": self.signal_args["t"]
            })
        self.signal_args["folder"] = self.exp_dir / "train"
        self.signal_args["query_args"] = self.query_args
        return self.generate_signal(self.signal_args)

    def load_train_data_uniform(self):
        signal_args = self.signal_args.copy()
        query_args = self.query_args.copy()
        n_samples = query_args["num_subsample"] * (signal_args["q"] ** query_args["b"]) *\
                    query_args["num_repeat"] * (signal_args["n"] + 1)
        query_args = {"subsampling_method": "uniform", "n_samples": n_samples}
        signal_args["folder"] = self.exp_dir / "train_uniform"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_test_data(self):
        signal_args = self.signal_args.copy()
        (self.exp_dir / "test").mkdir(exist_ok=True)
        signal_args["query_args"] = {"subsampling_method": "uniform", "n_samples": self.test_args.get("n_samples")}
        signal_args["folder"] = self.exp_dir / "test"
        signal_args["noise_sd"] = 0
        return self.generate_signal(signal_args)

    def load_full_data(self):
        #   TODO: implement
        return None

    def compute_model(self, method, model_kwargs, report=False, verbosity=0):
        if method == "smt":
            return self._calculate_smt(model_kwargs, report, verbosity)
        elif method == "smt_coded":
            raise NotImplementedError()
        elif method == "smt_robust":
            return self._calculate_smt_robust(model_kwargs, report, verbosity)
        else:
            raise NotImplementedError()

    def test_model(self, method, **kwargs):
        if method == "smt" or method == "smt_coded" or method == "smt_robust":
            return self._test_qary(**kwargs)
        else:
            raise NotImplementedError()

    def _calculate_smt(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates SMT coefficients.
        """
        if verbosity >= 1:
            print("Estimating GWHT coefficients with QSFT")
        qsft = QSFT(
            reconstruct_method_source="identity",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"]
        )
        self.train_signal.noise_sd = model_kwargs["noise_sd"]
        out = qsft.transform(self.train_signal, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _calculate_smt_robust(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates SMT coefficients.
        """
        if verbosity >= 1:
            print("Estimating coefficients with Coded SMT")

        def source_decoder(D, y):
            dec, err, decode_success = decode_robust(D, y, norm_factor=1, solution=None)
            return dec, decode_success

        smt = QSFT(
            reconstruct_method_source="coded",
            reconstruct_method_channel="nso",
            source_decoder=source_decoder,
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"],
            noise_sd=self.train_signal.noise_sd
        )

        self.train_signal.noise_sd = model_kwargs["noise_sd"]
        out = smt.transform(self.train_signal, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=True, sort=True)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _test_qary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            # Test NMSE both in signal and mobius domain

            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            # Compute signal NMSE of the Mobius transform
            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                query_indices_qary_batch = np.array(dec_to_bin_vec(sample_idx_dec_batch, self.n)).T
                locs = np.array([np.array(coord) for coord in beta_keys]).T
                strengths = np.array(beta_values)
                y_hat.append(((((1 - query_indices_qary_batch) @ locs) == 0) + 0) @ strengths)
            samples = np.array(samples)
            y_hat = np.concatenate(y_hat)
            NMSE_signal = np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2

            # Compute NMSE within sparse Mobius
            denom = sum(val ** 2 for val in beta_values)

            recovered_locs = [bin_vec_to_dec(np.array(loc)) for loc in beta_keys]
            true_locs = [bin_vec_to_dec(self.test_signal.loc[:, i]) for i in range(len(self.test_signal.strengths))]
            num = 0
            for coord in set(recovered_locs).difference(set(true_locs)):
                num += beta_values[recovered_locs.index(coord)] ** 2
            for coord in set(true_locs).difference(set(recovered_locs)):
                num += self.test_signal.strengths[true_locs.index(coord)] ** 2
            for coord in set(recovered_locs).intersection(set(true_locs)):
                num += (beta_values[recovered_locs.index(coord)] - self.test_signal.strengths[true_locs.index(coord)]) ** 2

            NMSE_mobius = num / denom

            return NMSE_signal, NMSE_mobius
        else:
            return 1, 1

    def _test_binary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, 2, 2 * self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / 2)
                y_hat.append(H @ np.array(beta_values))

            # TODO: Write with an if clause
            y_hat = np.abs(np.concatenate(y_hat))

            return np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
        else:
            return 1
