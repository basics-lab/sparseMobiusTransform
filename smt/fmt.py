"""
This file computes the mobius transform of a signal x
"""
import numpy as np
import timeit
import mobiusmodule


def fmt_iterative(x):
    N = len(x)
    i = 2
    x[1::2] -= x[::2]
    while i < N:
        b = [j % (2*i) < i for j in range(N)]
        x[np.roll(b, i)] -= x[b]
        i *= 2
    return x


def fmt_recursive(x):
    N = len(x)
    if N % 2 != 0:
        raise TypeError("Input singnal must have length power of 2 greater than 1")
    else:
        if N == 2:
            out = np.array([x[0], x[1] - x[0]])
        else:
            tf0 = fmt_recursive(x[:N // 2])
            tf1 = fmt_recursive(x[N // 2:])
            out = np.concatenate((tf0, tf1 - tf0))
    return out

def ifmt_recursive(x):
    N = len(x)
    if N % 2 != 0:
        raise TypeError("Input signal must have length power of 2 greater than 1")
    else:
        if N == 2:
            out = np.array([x[0], x[0] + x[1]])
        else:
            tf0 = ifmt_recursive(x[:N // 2])
            tf1 = ifmt_recursive(x[N // 2:])
            out = np.concatenate((tf0, tf0 + tf1))
    return out

def naive_mt(x):
    N = len(x)
    mt_matrix = np.array([[1, 0],[-1, 1]])
    full_mt_matrix = np.array([1])
    while N // 2 > 0:
        full_mt_matrix = np.kron(mt_matrix, full_mt_matrix)
        N = N // 2
    return full_mt_matrix @ x


if __name__ == '__main__':
    n = 12
    N = 2 ** n
    x = np.random.rand(N)
    out2 = fmt_recursive(x)
    mobiusmodule.mobius(x)
    def naive_test():
        x = np.random.rand(N)
        naive_mt(x)

    def recursive_test():
        x = np.random.rand(N)
        fmt_recursive(x)

    def iterative_test():
        x = np.random.rand(N)
        fmt_iterative(x)

    def cpython_test():
        x = np.random.rand(N)
        mobiusmodule.mobius(x)

    n_runs = 100
    t1 = timeit.timeit("naive_test()", "from __main__ import naive_test", number=n_runs)
    print(f"Naive implementation {t1/n_runs}")
    t2 = timeit.timeit("recursive_test()", "from __main__ import recursive_test", number=n_runs)
    print(f"Recursive implementation {t2/n_runs}")
    t3 = timeit.timeit("iterative_test()", "from __main__ import iterative_test", number=n_runs)
    print(f"Iterative implementation {t3/n_runs}")
    t4 = timeit.timeit("cpython_test()", "from __main__ import cpython_test", number=n_runs)
    print(f"Cpython implementation {t4/n_runs}")
    #out1 = naive_mt(x)
    #out2 = fmt_recursive(x)
    #out3 = fmt_iterative(x)
