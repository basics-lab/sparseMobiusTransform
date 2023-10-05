'''
Utility functions.
'''
import numpy as np
import scipy.fft as fft
import itertools
import math
import time
from scipy.spatial import ConvexHull
import zlib
import pickle
import json

def mt_tensored(x,n):
    """TODO"""
    pass

def imt_tensored(x,n):
    """TODO"""
    pass

def bin_to_dec(x):
    n = len(x)
    c = 2**(np.arange(n)[::-1])
    return c.dot(x).astype(np.int)

def nth_roots_unity(n):
    return np.exp(-2j * np.pi / n * np.arange(n))

def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)

def binary_ints(m):
    '''
    Returns a matrix where row 'i' is dec_to_bin(i, m), for i from 0 to 2 ** m - 1.
    From https://stackoverflow.com/questions/28111051/create-a-matrix-of-binary-representation-of-numbers-in-python.
    '''
    a = np.arange(2 ** m, dtype=int)[np.newaxis,:]
    b = np.arange(m, dtype=int)[::-1,np.newaxis]
    return np.array(a & 2**b > 0, dtype=int)

def comb(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def sign(x):
    '''
    Replacement for np.sign that matches the convention (footnote 2 on page 11).
    '''
    return (1 - np.sign(x)) // 2

def flip(x):
    '''
    Flip all bits in the binary array x.
    '''
    return np.bitwise_xor(x, 1)

def random_signal_strength_model(sparsity, a, b):
    magnitude = np.random.uniform(a, b, sparsity)
    phase = np.random.uniform(0, 2*np.pi, sparsity)
    return magnitude * np.exp(1j*phase)

def best_convex_underestimator(points):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    first_point_idx = np.argmin(vertices[:, 0])
    last_point_idx = np.argmax(vertices[:, 0])

    if last_point_idx == vertices.shape[0]:
        return vertices[first_point_idx:]
    if first_point_idx < last_point_idx:
        return vertices[first_point_idx:last_point_idx+1]
    else:
        return np.concatenate((vertices[first_point_idx:], vertices[:last_point_idx+1]))


def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))

def load_data(filename):
    start = time.time()
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def calc_hamming_weight():
    return None
