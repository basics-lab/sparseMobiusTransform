import numpy as np
import mobiusmodule

COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day_of_week', 'month', 'duration', 'campaign',
       'pdays', 'previous', 'poutcome']
def find_top_magnitude_indices(mt, num_kept):
    '''
    Find the indices of the top kept_amt percent of the magnitudes in mt
    '''
    indices = np.argsort(np.abs(mt))[-num_kept:]
    return indices

def find_top_weighted_magnitude_indices(allocs, mt, num_kept):
    '''
    Find the indices of the top kept_amt percent of the magnitudes in mt
    '''
    weights = 2 ** (allocs.shape[0] - np.sum(allocs, axis=0))
    indices = np.argsort(weights * np.abs(mt))[-num_kept:]
    return indices

def find_degree_d_indices(allocs, d):
    '''
    Find the indices of the top kept_amt percent of the magnitudes in mt
    '''
    degrees = np.sum(allocs, axis=0)
    return np.arange(degrees.shape[0])[degrees <= d]

def find_highest_magnitude_deg_d_indices(allocs, mt, d):
    '''
    Find the indices of the top kept_amt percent of the magnitudes in mt
    '''
    degrees = np.sum(allocs, axis=0)
    indices = np.arange(degrees.shape[0])[degrees == d]
    return indices[np.argsort(np.abs(mt[indices]))[::-1]]

def compute_NMSE(true_signal, recovered_signal):
    '''
    Compute the normalized mean squared error between the true signal and the recovered signal
    '''
    return np.sum((true_signal - recovered_signal) ** 2) / np.sum(true_signal ** 2)



if __name__ == '__main__':

    '''
    Generate subsets
    '''
    n = 16
    N = 2 ** n
    a = np.arange(N, dtype=int)[np.newaxis, :]
    b = np.arange(n, dtype=int)[::-1, np.newaxis]
    allocations = np.array(a & 2 ** b > 0, dtype=int)
    print(allocations[:,:10])

    '''
    Load model accuracies
    '''
    with open('feature_selection_accs.npy', 'rb') as f:
        model_accs = np.load(f)
    model_accs[0] = 0.5

    '''
    Compute mobius transform
    '''
    print(f'Model accuracies: {model_accs}')
    mt = np.copy(model_accs)
    mobiusmodule.mobius(mt)
    print(f'Mobius transform: {mt}')
    mobiusmodule.inversemobius(mt)
    print(f'Inverse mobius transform: {mt}')
    mobiusmodule.mobius(mt)

    nonzero_mt = np.nonzero(mt)[0]
    print(f'Count true nonzero: {len(nonzero_mt)}')

    '''
    Sparse recovery
    '''

    '''
    Approach 1: Highest magnitude
    '''
    print()
    print('Evaluating keeping highest magnitude')
    for num_kept in range(100, 501, 100):
        indices = find_top_magnitude_indices(mt, num_kept)
        sparse_mt = np.zeros(N)
        for ind in indices:
            sparse_mt[ind] = mt[ind]
        mobiusmodule.inversemobius(sparse_mt)
        print(f'NMSE for keeping top {num_kept} Mobius coefficients: {compute_NMSE(model_accs, sparse_mt)}')


    '''
    Approach 2: Weighted magnitude
    '''
    print()
    print('Evaluating keeping weighted magnitude')
    for num_kept in range(100, 501, 100):
        indices = find_top_weighted_magnitude_indices(allocations, mt, num_kept)
        sparse_mt = np.zeros(N)
        for ind in indices:
            sparse_mt[ind] = mt[ind]
        mobiusmodule.inversemobius(sparse_mt)
        print(f'NMSE for keeping top {num_kept} weighted Mobius coefficients: {compute_NMSE(model_accs, sparse_mt)}')

    '''
    Approach 3: Up to degree d
    '''
    print()
    print('Evaluating up to degree d')
    for deg in range(0,17):
        indices = find_degree_d_indices(allocations, deg)
        sparse_mt = np.zeros(N)
        for ind in indices:
            sparse_mt[ind] = mt[ind]
        mobiusmodule.inversemobius(sparse_mt)
        print(f'NMSE for up to degree {deg} Mobius coefficients: {compute_NMSE(model_accs, sparse_mt)}')

    '''
    Approach 4: Highest ranking elements for degree d
    '''
    print()
    print('Evaluating up to degree d')
    for deg in range(1,4):
        indices = find_highest_magnitude_deg_d_indices(allocations, mt, deg)
        print(f'Degree {deg}')

        for ind in indices[:10]:
            cols = []
            for i in range(allocations.shape[0]):
                if allocations[i,ind] == 1:
                    cols.append(COLUMNS[i])
            print(cols)
            print(mt[ind])
            print(model_accs[ind])
        print()