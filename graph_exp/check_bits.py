import numpy as np

def dec_to_bin_vec(x, n):
    qary_vec = []
    x = np.array(x, dtype=object)
    for i in range(n):
        qary_vec.append(np.array([a // (2 ** (n - (i + 1))) for a in x], dtype=object))
        x = x - (2 ** (n-(i + 1))) * qary_vec[i]
    return np.array(qary_vec, dtype=int)

def fast_dec_to_bin_vec(x, n):
    n_bytes = (n + 7) // 8
    # Convert to bytes
    byte_list = [int(val).to_bytes(n_bytes, byteorder='big') for val in x]
    all_bytes = b''.join(byte_list)
    uint8_arr = np.frombuffer(all_bytes, dtype=np.uint8)
    bits = np.unpackbits(uint8_arr)
    bits = bits.reshape(len(x), -1)
    
    # bits has size n_bytes * 8.
    # 'big' endian means MSB is at index 0 of the bytes.
    # But if n < n_bytes * 8, we have padding zeros at the START (since it's big endian number).
    # e.g. x=1, n=10. n_bytes=2 (16 bits).
    # bytes: 00000000 00000001.
    # bits: 00000000 00000001.
    # We want last n bits?
    # dec_to_bin_vec(1, 10) -> [0,0,0,0,0,0,0,0,0,1].
    # So we want the last n bits of the row.
    
    return bits[:, -n:].T

# Test
n = 10
vals = [1, 2**9 + 1, 255]
print("Testing n=", n, "vals=", vals)

slow = dec_to_bin_vec(vals, n)
print("Slow:\n", slow)

fast = fast_dec_to_bin_vec(vals, n)
print("Fast:\n", fast)

assert np.array_equal(slow, fast)
print("Matched!")

n = 20
vals = [2**19, 12345]
print("Testing n=", n)
slow = dec_to_bin_vec(vals, n)
fast = fast_dec_to_bin_vec(vals, n)
assert np.array_equal(slow, fast)
print("Matched!")
