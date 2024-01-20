from auc_exp.auc_src.arbitrary import Arbitrary
from auc_exp.auc_src.matching import Matching
from auc_exp.auc_src.paths import Paths
from auc_exp.auc_src.proximity import Proximity
from auc_exp.auc_src.scheduling import Scheduling
from smt.input_signal_subsampled import SubsampledSignal
import numpy as np
from multiprocessing import Pool
from smt.utils import dec_to_bin_vec
from tqdm import tqdm


class AuctionSubsampledSignal(SubsampledSignal):
    def __init__(self, **kwargs):
        setting = kwargs.get("setting")
        assert setting in {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}, \
            "Setting must be one of {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}"
        seed = kwargs.get("seed")
        self.setting = eval(setting.capitalize())(seed)
        self.n = self.setting.num_items
        self.q = 2
        kwargs['n'] = self.n
        kwargs['n_runs'] = 100
        kwargs['q'] = self.q
        self.noise_sd = 0
        self.pool = Pool()

        super().__init__(**kwargs)

        # self.pool.close()

    def subsample(self, query_indices):

        batch_size = 250
        res = np.zeros(len(query_indices))
        counter = 0

        query_batches = np.array_split(query_indices, len(query_indices)//batch_size)
        pbar = tqdm(desc='Evaluating Auction Value Function', total=len(query_batches))
        for new_res in self.pool.imap(self.sampling_function, query_batches):
            res[counter: counter+len(new_res)] = new_res
            counter += len(new_res)
            pbar.update(1)

        return res

    def sampling_function(self, query_batch):
        x = np.array(dec_to_bin_vec(query_batch, self.n))
        values = np.zeros(len(query_batch))
        for i in range(len(query_batch)):
            values[i] = self.setting.value_function(0, x[:, i])
        return values

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
