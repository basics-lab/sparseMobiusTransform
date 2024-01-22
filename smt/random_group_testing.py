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
    Does robust decoding of the group testing model.

    Parameters
    ----------
    A : The matrix to use
    y : The observed test results
    norm_factor : The normalization factor to use
    solution : The correct answer

    Returns
    -------
    dec : the decoder output
    err : dec - solution
    decode_success : err == 0
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
    """

    Parameters
    ----------
    A : Matrix to be tested
    t : abstractly (Number of defects)
    n_runs : Number of tests to run
    kwargs : TODO

    Returns
    -------
    A list of the fraction of [passed, failed, detected] tests.
    """
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
    """

    Parameters
    ----------
    m : number of rows of group testing matrix
    wt : weight parameter for random matrix generation
    n : number of columns in the group testing matrix
    t : number of defects
    kwargs : n_matrix (number of test matrices), n_runs (number of times to run each test matrix)

    Returns
    -------
    acc : an array containing the fraction of (successes, fails, detected fails)
    """
    n_mtrx = kwargs.get("n_mtrx", 10)
    n_runs = kwargs.get("n_runs", 100)
    mtrx_type = kwargs.get("mtrx_type", "bernoulli")
    kwargs.pop("n_runs")
    for i in range(n_mtrx):
        if mtrx_type == "bernoulli":
            A = get_random_bernoulli_matrix(n, m, wt/t)
        elif mtrx_type == "const_col":
            A = get_random_near_const_weight_mtrx(n, m, int(wt*m/t))
        else:
            raise ValueError("Matrix type is not implemented")
        if i == 0:
            acc = test_design(A, t, n_runs, **kwargs)
        else:
            acc += test_design(A, t, n_runs, **kwargs)
    return acc / n_mtrx


def test_wt_range(m, n, t, min_wt=None, max_wt=None, **kwargs):
    """

    Parameters
    ----------
    m : Number of rows in the group testing matrix
    n : Number of columns in the group testing matrix
    t : Number of defective items in model
    min_wt : Minimum weight parameter in search range (default=0.4)
    max_wt : Maximum weight parameter in search range (default=1.2)
    kwargs

    Returns
    -------
    acc_list : includes the number of successes, fails and detected fails in decoding for each weight in the range(
    min_wt, max_wt) inclusive
    """
    verbose = kwargs.get("verbose", False)
    mtrx_type = kwargs.get("mtrx_type", "bernoulli")
    if min_wt is None:
        min_wt = 0.4
    min_val = int((min_wt * m) // t)
    if max_wt is None:
        max_wt = 1.2
    max_val = int((max_wt * m) // t) + 1
    n_points = max_val - min_val
    if mtrx_type == "bernoulli":
        wt_range = np.linspace(max_wt, min_wt, n_points)
    elif mtrx_type == "const_col":
        wt_range = np.array([range(min_val, max_val)])*t/m
    acc_list = np.zeros((n_points, 3))  # 3 because (success, fails, detected fails)
    if verbose:
        print("Computing the optimal column weight for group testing design:")
        print(f"searching in [{min_wt},{max_wt}]")
    for i in range(n_points):
        acc_list[i, :] = test_weight(m, wt_range[i], n, t, **kwargs)
        if verbose:
            print(f"Finished wt={wt_range[i]}")
    if verbose:
        print(acc_list)
        top_idx = np.argmax(acc_list[:, 0])
        acc = acc_list[top_idx, 0]
        if acc > 0.9:
            print(f"Max accuracy is {acc}, when the weight is {wt_range[top_idx]}.")
        else:
            print(f"Max accuracy is {acc}, since it is too low, we will not continue. Choose higher m.")
    return acc_list


def get_gt_delay_matrix(n, m, wt, t):
    """
    In the limit, all the group testing matrices are the same, but in this finite range, there are some that might be
    better than others. This code will search through a bunch of random ones and find the best one.
    Parameters
    ----------
    n : Number of rows in the group testing matrix
    m : Number of rows in the group testing matrix
    wt : weight parameter of the group testing matrix
    t : number of defects

    Returns
    -------
    ret_A : A matrix of size (m,n) which is deemed to be the best matrix for group testing.
    """
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
    """
    Gets a group testing matrix to use as the M matrix

    Parameters
    ----------
    n : number of columns of full group testing matrix
    m : number of rows of full group testing matrix
    b : number of rows to be taken to construct the matrix M
    wt : the weight parameter to use

    Returns
    -------
    M : A group testing matrix randomly generated
    """
    M = get_random_near_const_weight_mtrx(n, m, wt)
    return M[:b, :]


def random_deg_t_vecs(t, n, num):
    test_vecs = np.random.randint(n, size=(t, num))
    x = np.zeros((n, num))
    for i in range(num):
        x[test_vecs[:, i].astype(int), i] = 1
    return x


def test_uniformity(A, sample_model, num):
    x = sample_model(num)
    hashed_vals = (A @ x > 0)
    sample_mean = np.sum(hashed_vals, 1)/num
    sample_mean = sample_mean[:, np.newaxis]
    hashed_vals = hashed_vals - sample_mean
    sample_cov = (hashed_vals @ hashed_vals.T)/(num-1)
    rows = A.shape[0]
    mean_err = np.linalg.norm(sample_mean - 0.5*np.ones_like(sample_mean))/rows
    cov_err = np.linalg.norm(sample_cov - 0.25*np.eye(rows))/(rows**2)
    return mean_err, cov_err


def plot_vs_m(n, t, **kwargs):
    """
    This code will plot the success/failure curve of a group testing procedure, operating in two main modes
    1. if fixed_wt_prob is passed as a keyword argument, it will use that fixed parameter for all M
    2. if fixed_wt_prob is not passed, it will search over a range of test weights.

    Parameters
    ----------
    n : number of columns in group testing
    t : number of defects
    kwargs : test_multiple  - searches over a range of normalization parameters
             debug          - disables parallel processing

    Returns
    -------
    Nothing
    """
    # Parse Inputs
    m_range = kwargs.get("m_range")
    fixed_wt = kwargs.get("fixed_wt_prob", False)
    mtrx_type = kwargs.get("mtrx_type", "bernoulli")
    min_wt = kwargs.get('min_wt', 0.4)
    max_wt = kwargs.get('max_wt', 1.2)
    extra_text = kwargs.get("extra_text", "")
    debug = kwargs.get("debug", False)
    test_multiple = kwargs.get("test_multiple", False)
    robust = kwargs.get("robust", False)
    if m_range is None:
        arguments = list(range(5, 45, 2))
    else:
        if len(m_range) == 2:
            arguments = list(range(m_range[0], m_range[1], 2))
        elif len(m_range) == 3:
            arguments = list(range(m_range[0], m_range[1], m_range[2]))
        else:
            raise ValueError("m_range should have length 2 or 3")
    n_points = len(arguments)
    if (not fixed_wt) and test_multiple:
        raise ValueError("Can't run non fixed_wt and test_multiple at the same time")
    if test_multiple and not robust:
        raise ValueError("test_multiple is meant to be used with the robust parameter")
    # Evaluate Testing
    max_processes = multiprocessing.cpu_count()
    if fixed_wt:
        run_func = partial(test_weight, wt=fixed_wt, n=n, t=t, **kwargs)
    else:
        run_func = partial(test_wt_range, n=n, t=t, min_wt=min_wt, max_wt=max_wt, **kwargs)
    if debug:
        results = [run_func(x) for x in arguments]
    else:
        # Create a multiprocessing pool with the maximum number of processes
        with multiprocessing.Pool(processes=max_processes) as pool:
            results = pool.map(run_func, arguments)

    # Print the results
    extra_text = extra_text + f" Fixed Weight={fixed_wt:.2f}" if fixed_wt else extra_text
    n_figs = 2 if (fixed_wt and not test_multiple) else 3
    plt.suptitle(f"t={t}, n={n}" + " " + extra_text)
    if fixed_wt:
        if test_multiple:
            norms = kwargs.get('norms')
            plots = np.zeros((3, n_points))
            idx_array = []
            for i in range(n_points):  # We need to search over all normalization factors to find the best one
                top_idx = np.argmax(results[i][0, :])
                plots[:, i] = results[i][:, top_idx]
                idx_array.append(top_idx)
        else:
            plots = np.array(results).T  # The three rows are just what we need to plot
    else:
        plots = np.zeros((4, n_points))
        for i in range(n_points):
            top_idx = np.argmax(results[i][:, 0])
            plots[:3, i] = results[i][top_idx, :]
            min_val = int((min_wt * arguments[i]) // t)
            max_val = int((max_wt * arguments[i]) // t) + 1
            n_points = max_val - min_val
            top_wt_param = np.linspace(max_wt, min_wt, n_points)[top_idx]
            plots[-1, i] = top_wt_param
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
    if not fixed_wt:
        plt.subplot(n_figs, 1, 3)
        plt.plot(arguments, plots[3, :])
        plt.xlabel('m')
        plt.ylabel('Opt. Column Weight')
    elif test_multiple:
        plt.subplot(n_figs, 1, 3)
        plt.plot(arguments, [norms[i] for i in idx_array])
        plt.xlabel('m')
        plt.ylabel('Best Normalization')
    plt.show()


if __name__ == "__main__":
    """
    In this section, we conduct a series of tests. Before major modifications to this code, all of these tests should be
    run to ensure that all functionality of this code is working. 
    """
    example_number = 7
    if example_number == 1:  # Test robust with fixed wt, and multiple norms
        norms = [0.3, 0.6, 0.8, 1, 2]
        n = 500
        p = 0.05
        t = 10
        acc = plot_vs_m(n=n,
                        t=t,
                        robust=True,
                        fixed_wt_prob=np.log(2),
                        test_multiple=True,
                        norms=norms,
                        n_runs=100,
                        n_mtrx=10,
                        m_range=(50, 400, 50),
                        p=p)
    elif example_number == 2:  # Test robust with fixed wt and a single norm
        n = 500
        p = 0.05
        t = 10
        acc = plot_vs_m(n=n,
                        t=t,
                        robust=True,
                        fixed_wt_prob=np.log(2),
                        n_runs=100,
                        n_mtrx=10,
                        m_range=(50, 400, 50),
                        p=p)
    elif example_number == 3:  # Test non-robust with fixed wt and single norm
        n = 500
        t = 10
        acc = plot_vs_m(n=n,
                        t=t,
                        fixed_wt_prob=np.log(2),
                        n_runs=100,
                        n_mtrx=10,
                        m_range=(50, 200, 10))
    elif example_number == 4:  # Test non-robust with weight range
        n = 500
        t = 10
        acc = plot_vs_m(n=n,
                        t=t,
                        n_runs=100,
                        n_mtrx=10,
                        m_range=(50, 200, 10))
    elif example_number == 5:  # Test robust with weight range (This test is slow)
        n = 500
        t = 6
        p = 0.05
        acc = plot_vs_m(n=n,
                        t=t,
                        robust=True,
                        n_runs=100,
                        n_mtrx=10,
                        m_range=(50, 200, 50),
                        p=p)
    elif example_number == 6:  # Test non-robust with fixed wt, using const. col weight matrix
        n = 500
        t = 10
        acc = plot_vs_m(n=n,
                        t=t,
                        fixed_wt_prob=np.log(2),
                        n_runs=100,
                        n_mtrx=10,
                        mtrx_type="const_col",
                        m_range=(50, 200, 10))
    elif example_number == 7:
        n = 500
        m = 140
        t = 10
        b = 10
        n_mc = 50000
        A = get_random_near_const_weight_mtrx(n, m, int(np.log(2)*m/t))
        mean_err, cov_err = test_uniformity(A[:b, :], lambda x: random_deg_t_vecs(t, n, x), n_mc)
        print(f"Normalized Mean L2 ={mean_err}\nNormalized Cov L2 = {cov_err}")
        A = get_random_bernoulli_matrix(n, m, np.log(2)/t)
        mean_err, cov_err = test_uniformity(A[:b, :], lambda x: random_deg_t_vecs(t, n, x), n_mc)
        print(f"Normalized Mean L2 ={mean_err}\nNormalized Cov L2 = {cov_err}")
