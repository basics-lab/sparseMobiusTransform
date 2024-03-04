import numpy as np
import sys
from tqdm import tqdm
from auc_src.arbitrary import Arbitrary
from auc_src.matching import Matching
from auc_src.paths import Paths
from auc_src.proximity import Proximity
from auc_src.scheduling import Scheduling

if __name__ == '__main__':
    SETTING = sys.argv[1]
    for SEED in tqdm(range(100)):
        setting = eval(SETTING.capitalize())(SEED, regime='small')
        n = setting.num_items
        N = 2 ** n

        a = np.arange(N, dtype=int)[np.newaxis, :]
        b = np.arange(n, dtype=int)[::-1, np.newaxis]
        allocations = np.array(a & 2 ** b > 0, dtype=int)

        values = np.zeros(N)
        for i in range(N):
            values[i] = setting.value_function(0, allocations[:,i])

        with open(f'saved_values2/{SETTING}_{SEED}', 'wb') as f:
            np.save(f, values)
