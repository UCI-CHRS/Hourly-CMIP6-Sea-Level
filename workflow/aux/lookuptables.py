
"""Creates fast lookup tables for use with pre-computed values 
compatible with Numba.
"""

import numpy as np
from numba import njit

@njit
def make_keys(A, B, C, scale=1000.0):
    n = A.size
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        ai = int(round(A[i] * scale))
        bi = int(round(B[i] * scale))
        ci = int(round(C[i] * scale))
        keys[i] = (ai * 1_000_000_000) + (bi * 1_000_000) + ci
    return keys


"""
keys = make_keys(A, B, C, scale=1000.0)
order = np.argsort(keys)
keys = keys[order]
D = D[order]
"""

@njit
def lookup_many(Aq, Bq, Cq, keys, D, scale=1000.0):
    out = np.empty(Aq.size, dtype=D.dtype)
    n = keys.size
    for j in range(Aq.size):
        a = int(round(Aq[j] * scale))
        b = int(round(Bq[j] * scale))
        c = int(round(Cq[j] * scale))
        target = a * 1_000_000_000 + b * 1_000_000 + c

        # binary search
        lo, hi = 0, n - 1
        val = np.nan
        while lo <= hi:
            mid = (lo + hi) // 2
            if keys[mid] == target:
                val = D[mid]
                break
            elif keys[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        out[j] = val
    return out

"""
Aq = np.array([1.1, 2.0, 3.3])
Bq = np.array([0.5, 0.0, 0.2])
Cq = np.array([0.0, 0.5, 0.1])

result = lookup_many(Aq, Bq, Cq, keys, D)

"""
