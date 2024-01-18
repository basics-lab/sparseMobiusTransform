
from auc_exp.auc_src.arbitrary import Arbitrary
from auc_exp.auc_src.matching import Matching
from auc_exp.auc_src.paths import Paths
from auc_exp.auc_src.proximity import Proximity
from auc_exp.auc_src.scheduling import Scheduling
from smt.input_signal_subsampled import SubsampledSignal

from multiprocessing import Pool

class AuctionSubsampledSignal(SubsampledSignal):
    def __init__(self, **kwargs):
        setting = kwargs.get("setting")
        assert setting in {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}, \
            "Setting must be one of {'arbitrary', 'matching', 'paths', 'proximity', 'scheduling'}"
        seed = kwargs.get("seed")
        self.setting = eval(setting.capitalize())(seed)

        self.pool = Pool()

        super().__init__(**kwargs)

        self.pool.close()

    def subsample(self, query_indices):

        batch_size = 250
        res = np.zeros(len(query_indices))
        counter = 0

        query_batches = np.array_split(query_indices, len(query_indices)//batch_size)

        for new_res in self.pool.imap(sampling_function, query_batches):
            res[counter: counter+len(new_res)] = new_res
            counter += len(new_res)
            # pbar.update(len(new_res))

        return res
