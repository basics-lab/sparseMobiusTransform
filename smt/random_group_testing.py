import numpy as np
from scipy.optimize import linprog
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import math

def get_random_near_const_weight_mtrx(n: int, m: int, wt):
    """

    Parameters
    ----------
    n : number of cols
    m : number of rows
    wt :

    Returns
    -------
    A near-constant column-weight matrix of size (n,m), where wt coordinates in each row are set to 1, sampling with
    replacement
    """
    nz_vals = np.random.randint(m, size=(wt, n))
    A = np.zeros((m, n))
    for i in range(n):
        A[nz_vals[:, i], i] = 1
    return A.astype(int)

def get_random_bernoulli_matrix(n: int, m: int, prob):
    """

    Parameters
    ----------
    n : number of cols
    m : number of rows
    wt :

    Returns
    -------
    A bernoulli IID matrix of size (n,m), where wt coordinates in each row are set to 1, sampling with
    replacement
    """
    A = np.random.rand(m, n) < prob
    return A.astype(int)

def decode(A, y):
    """
    Parameters
    ----------
    A :  binary matrix,
    y : boolean ndarray with shape (n,1)
    """
    m, n = A.shape
    c = np.ones(n)
    A_ub = -A[y[:, 0], :]
    A_eq = A[np.invert(y[:, 0]), :]
    n_pos = A_ub.shape[0]
    b_ub = -np.ones(n_pos)
    b_eq = np.zeros(m - n_pos)
    bounds = (0, None)
    x = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).x
    if x is None:
        x = c  # (This should never happen in the noiseless case, but it seems to be happening? bug?)
        decode_success = False
    else:
        decode_success = ((x == 0) + (x == 1)).all()
    return x.astype(int), decode_success


def decode_robust(A, y, norm_factor, solution):
    """

    Parameters
    ----------
    A
    y

    Returns
    -------

    """
    verbose = False
    options = {}
    tol = 1.0e-8
    options["tol"] = tol
    m, n = A.shape

    # Objective
    c = np.ones(n + m)
    c[n:] *= norm_factor

    # Inequality constraint
    A_ub = -A[y[:, 0], :]
    n_pos = A_ub.shape[0]
    B_ub = np.zeros((m, m))
    B_ub[y[:, 0], y[:, 0]] = np.ones(n_pos)
    B_ub = -B_ub[y[:, 0], :]
    A_ub = np.hstack((A_ub, B_ub))
    b_ub = -np.ones(n_pos)

    # Box constraints
    bounds = [(0, 1)]*n + [(0, 1) if y[i, 0] else (0, None) for i in range(m)]

    # Equality Constraint
    A_eq = A[np.invert(y[:, 0]), :]
    B_eq = np.zeros((m, m))
    B_eq[np.invert(y[:, 0]), np.invert(y[:, 0])] = np.ones(m - n_pos)
    B_eq = -B_eq[np.invert(y[:, 0]), :]
    A_eq = np.hstack((A_eq, B_eq))
    b_eq = np.zeros(m - n_pos)

    # Error Checking
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    x = res.x
    if verbose:
        print(f"Opt Found: f={res.fun}, Opt. Status={res.status}")
        print(f"True Desired Solution f={c.T @ solution[:, 0]}")
        print(f"Inequality Constraint Satisfied: {np.all(b_ub - (A_ub @ solution[:, 0]) >= 0)}")
        print(f"Equality Constraint Satisfied: {np.all(np.abs(b_eq - A_eq @ solution[:, 0]) < tol)}")
    if x is None:
        x = c  # (This should never happen in the noiseless case, but it seems to be happening? bug?)
        decode_success = False
    else:
        decode_success = (np.abs(x - x.round()) < tol).all()  # Check if result is binary
    x = x.astype(int)
    dec = x[:n]
    err = x[n:]
    return dec, err, decode_success


def decode_robust_multiple(A, y, norms, solution):
    f = lambda x: decode_robust(A, y, x, solution)
    return [f(x) for x in norms]


def test_design(A, t, n_runs, **kwargs):
    # Parse Arguments
    verbose = kwargs.get("verbose", False)
    early_exit_rate = kwargs.get("early_exit_rate", 20)
    m, n = A.shape
    data_model = kwargs.get("data_model")
    robust_decode = kwargs.get("robust", False)
    test_multiple = kwargs.get("test_multiple", False)
    norms = kwargs.get("norms", [1])
    num_norms = len(norms)
    norm_factor = kwargs.get("norm_factor", 1)
    p = kwargs.get("p", 0)
    # variable init
    if num_norms == 1:
        passed = 0
        failed = 0
        detected = 0
    else:
        passed = [0 for _ in range(num_norms)]
        failed = [0 for _ in range(num_norms)]
        detected = [0 for _ in range(num_norms)]
    # Pre-compute the error patterns
    if data_model is None:
        test_vecs = np.random.randint(n, size=(t, n_runs))
    for i in range(n_runs):

        # Get data sample
        if data_model is None:
            x = np.zeros((n, 1))
            x[test_vecs[:, i]] = 1
        else:
            x = data_model()

        # Decode
        if robust_decode:
            err = np.random.rand(m, 1) > (1 - p)
            sig = A @ x
            y = (sig > 0) ^ err
            error_sig = sig * err - ((sig > 0) * err) + err
            if test_multiple:
                res = decode_robust_multiple(A, y, norms, np.concatenate((x, error_sig)))
            else:
                x_hat, _, success = decode_robust(A, y, norm_factor, np.concatenate((x, error_sig)))
        else:
            y = (A @ x) > 0
            x_hat, success = decode(A, y)

        # Collect Statistics
        if num_norms == 1:
            if success:
                if np.all(x[:, 0] == x_hat):
                    passed += 1
                else:
                    failed += 1
            else:
                detected += 1
            if (i % early_exit_rate) == 0 and passed < (i // 10):  # early exit if passed < 10%
                # 10% after 10
                # runs
                if verbose:
                    print("Aborting test, error is too high")
                return np.array([passed, failed, detected]) / (i + 1)
        else:
            for j in range(num_norms):
                x_hat, _, success = res[j]
                if success:
                    if np.all(x[:, 0] == x_hat):
                        passed[j] += 1
                    else:
                        failed[j] += 1
                else:
                    detected[j] += 1
                # Should implement early exit if all successes are too low
    return np.array([passed, failed, detected]) / n_runs


def test_weight(m, wt, n, t, **kwargs):
    n_mtrx = kwargs.get("n_mtrx", 10)
    n_runs = kwargs.get("n_runs", 100)
    kwargs.pop("n_runs")
    wt_is_prob = kwargs.get("fixed_wt_prob", False)
    if wt_is_prob:
        wt = wt / t
        print(wt)
    for i in range(n_mtrx):
        #A = get_random_near_const_weight_mtrx(n, m, wt)
        A = get_random_bernoulli_matrix(n, m, wt)
        if i == 0:
            acc = test_design(A, t, n_runs, **kwargs)
        else:
            acc += test_design(A, t, n_runs, **kwargs)
    return acc / n_mtrx


def test_wt_range(m, n, t, min_wt=None, max_wt=None, **kwargs):
    verbose = kwargs.get("verbose", False)
    if min_wt is None:
        nu_min = 0.4
        min_wt = int((nu_min * m) // t)
    if max_wt is None:
        nu_max = 1.2
        max_wt = int((nu_max * m) // t) + 1
    acc_list = np.zeros((max_wt - min_wt, 3))
    if verbose:
        print("Computing the optimal column weight for group testing design:")
        print(f"searching in [{min_wt},{max_wt - 1}]")
    for i in range(max_wt - min_wt):
        acc_list[i, :] = test_weight(m, min_wt + i, n, t, **kwargs)
        if verbose:
            print(f"Finished wt={min_wt + i}")
    if verbose:
        print(acc_list)
        top_idx = np.argmax(acc_list[:, 0])
        acc = acc_list[top_idx, 0]
        if acc > 0.9:
            print(f"Max accuracy is {acc}, when the weight is {min_wt + top_idx}.")
        else:
            print(f"Max accuracy is {acc}, since it is too low, we will not continue. Choose higher m.")
    return acc_list


def get_gt_delay_matrix(n, m, wt, t):
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
    zero_delays = np.zeros(n, )
    ret_A = np.vstack((zero_delays, ret_A))
    return ret_A


def get_gt_M_matrix(n, m, b, wt):
    M = get_random_near_const_weight_mtrx(n, m, wt)
    return M[:b, :]


def plot_vs_m(n, t, **kwargs):
    # Parse Inputs
    m_range = kwargs.get("m_range")
    fixed_wt = kwargs.get("fixed_wt_prob", False)
    extra_text = kwargs.get("extra_text", "")
    debug = kwargs.get("debug", False)
    test_multiple = kwargs.get("test_multiple", False)
    if m_range is None:
        arguments = list(range(5, 45, 2))
    else:
        if len(m_range) == 2:
            arguments = list(range(m_range[0], m_range[1], 2))
        elif len(m_range) == 3:
            arguments = list(range(m_range[0], m_range[1], m_range[2]))
        else:
            raise ValueError("M should have length 2 or 3")
    n_points = len(arguments)
    if (not fixed_wt) and test_multiple:
        raise ValueError("Can't run fixed_wt and test_multiple at the same time")
    # Evaluate Testing
    max_processes = multiprocessing.cpu_count()
    if fixed_wt:
        run_func = partial(test_weight, wt=fixed_wt, n=n, t=t, **kwargs)
    else:
        run_func = partial(test_wt_range, n=n, t=t, **kwargs)

    if debug:
        results = [run_func(x) for x in arguments]
    else:
        # Create a multiprocessing pool with the maximum number of processes
        with multiprocessing.Pool(processes=max_processes) as pool:
            results = pool.map(run_func, arguments)

    # Print the results
    extra_text = extra_text + f" Fixed Weight={fixed_wt}" if fixed_wt else extra_text
    n_figs = 2 if fixed_wt else 3
    plt.suptitle(f"t={t}, n={n}" + " " + extra_text)
    if fixed_wt:
        if test_multiple:
            plots = np.zeros((3, n_points))
            for i in range(n_points):
                top_idx = np.argmax(results[i][0, :])
                plots[:, i] = results[i][:, top_idx]
        else:
            plots = np.array(results).T
    else:
        plots = np.zeros((4, n_points))
        for i in range(n_points):
            top_idx = np.argmax(results[i][:, 0])
            plots[:3, i] = results[i][top_idx, :]
            plots[-1, i] = top_idx + int((0.4 * arguments[i]) // t)
    plt.subplot(n_figs, 1, 1)
    plt.plot(np.insert(arguments, 0, 0), np.insert(plots[0, :], 0, 0))
    plt.xlabel('m')
    plt.ylabel('P(Success)')
    plt.subplot(n_figs, 1, 2)
    plt.plot(arguments, plots[1, :] + plots[2, :])
    plt.plot(arguments, plots[2, :])
    plt.legend(['Failures', 'Detected Failures'])
    plt.xlabel('m')
    plt.ylabel('P(Fail)')
    if fixed_wt is False:
        plt.subplot(n_figs, 1, 3)
        plt.plot(arguments, plots[3, :])
        plt.xlabel('m')
        plt.ylabel('Opt. Column Weight')
    plt.show()
    # wt, acc = optimal_wt(n, m, t)

    #    raise ValueError("Sparsity level is too low, try a higher value.")
    # D = get_gt_delay_matrix(n, m, wt, t)
    # results = full_profile_design(D, t, 300)
    # M = get_gt_M_matrix(n, m, b, wt)


if __name__ == "__main__":
    n = 500
    p = 0.05
    t = 10
    wt = 7
    norms = [0.3, 0.6, 0.8, 1, 2]
    acc = plot_vs_m(n=n,
                    t=t,
                    robust=True,
                    test_multiple=True,
                    fixed_wt_prob=math.log(2),
                    norms=norms,
                    n_runs=100,
                    n_mtrx=10,
                    m_range=(50, 400, 50),
                    p=p)
    # A = get_random_near_const_weight_mtrx(n, m, 8)
    # nz = np.random.randint(n, size=(3, 1))
    # x = np.zeros((n, 1))
    # x[nz, :] = 1
    # err = np.random.rand(m, 1) > (1 - p)
    # y = (A @ x) > 0
    # r = y ^ err
    # dec, det_err, success = decode_robust(A, r)
    # err = err[:, 0].astype(int)
    # correct = x[:, 0].astype(int)
    # breakpoint()
