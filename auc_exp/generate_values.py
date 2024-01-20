import numpy as np
import os
import sys
from arbitrary import Arbitrary
from matching import Matching
from paths import Paths
from proximity import Proximity
from scheduling import Scheduling

if __name__ == '__main__':
    SETTING = sys.argv[1]
    for SEED in range(100):
        setting = eval(SETTING.capitalize())(SEED)
        n = setting.num_items
        N = 2 ** n

        a = np.arange(N, dtype=int)[np.newaxis, :]
        b = np.arange(n, dtype=int)[::-1, np.newaxis]
        allocations = np.array(a & 2 ** b > 0, dtype=int)
        print()

        values = np.zeros(N)
        for i in range(N):
            values[i] = setting.value_function(0, allocations[:,i])

        with open(f'/global/scratch/users/landonb/{SETTING}_{SEED}', 'wb') as f:
            np.save(f, values)
