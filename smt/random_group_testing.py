import numpy as np
from scipy.optimize import linprog


def get_random_near_const_weight_mtrx(n, m, wt):
    nz_vals = np.random.randint(m, size=(wt, n))
    A = np.zeros((m, n))
    for i in range(n):
        A[nz_vals[:, i], i] = 1
    return A.astype(int)


"""
A is a binary matrix,
y is a boolean ndarray with shape (n,1)
"""
def decode(A,y):
    m, n = A.shape
    c = np.ones(n)
    A_ub = -A[y[:,0], :]
    A_eq = A[np.invert(y[:, 0]), :]
    n_pos = A_ub.shape[0]
    b_ub  = -np.ones(n_pos)
    b_eq = np.zeros(m - n_pos)
    bounds = (0, None)
    return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).x.astype(int)


def test_weight(wt, m, n, t):
    n_runs = 10
    acc = 0
    for i in range(n_runs):
        A = get_random_near_const_weight_mtrx(n, m, wt)
        acc += test_design(A, t)
    return acc/n_runs


def test_design(A, t):
    m, n = A.shape
    n_runs = 200
    acc = 0
    test_vecs = np.random.randint(n, size=(t, n_runs))
    for i in range(n_runs):
        x = np.zeros((n,1))
        x[test_vecs[:, i]] = 1
        y = (A @ x) > 0
        x_hat = decode(A, y)
        if np.all(x[:, 0] == x_hat):
            acc += 1
    return acc/n_runs


def optimal_wt(n, m, t):
    nu_min = 0.4
    verbose = True
    nu_max = 1.2
    min_wt = int((nu_min * m) // t)
    max_wt = int((nu_max*m)//t)+1
    acc_list = np.zeros(max_wt - min_wt)
    if verbose:
        print("Computing the optimal column weight for group testing design:")
        print(f"searching in [{min_wt},{max_wt-1}]")
    for i in range(max_wt - min_wt):
        acc_list[i] = test_weight(min_wt + i, m, n, t)
        if verbose:
            print(f"Finished wt={min_wt + i}")
    i_opt = np.argmax(np.array(acc_list))
    return min_wt + i_opt, acc_list[i_opt]


def get_gt_delay_matrix(n, m, wt):
    # Now we compute a few different random matricies, and test a few
    n_candidates = 5
    ret_acc = 0
    ret_A = None
    for i in range(n_candidates):
        A = get_random_near_const_weight_mtrx(n, m, wt)
        acc = test_design(A, t)
        if acc > ret_acc:
            ret_A = A
            ret_acc = acc
    print(f"Among all the candidates, the one with the highest accuracy is {acc}, using that one.")
    return ret_A


def get_gt_M_matrix(n, m, b, wt):
    M = get_random_near_const_weight_mtrx(n, m, wt)
    return M[:b, :]


if __name__ == '__main__':
    t = 3
    n = 100
    m = 30
    b = 4
    wt, acc = optimal_wt(n, m, t)
    if acc > 0.9:
        print(f"Accuracy is {acc} when the weight is {wt}, since it is high enough, we will proceed.")
    else:
        print(f"Accuracy is {acc} in the best case, since it is too low, we will not continue. Choose higher m > {m}.")
        ValueError("Sparsity level is too low, try a higher value.")
    D = get_gt_delay_matrix(n, m, wt)
    M = get_gt_M_matrix(n, m, b, wt)
