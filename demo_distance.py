"""
Pairwise squared distances |x_i - y_j|^2  via the one-matmul method.

Method (TPU-friendly):
    |x_i - y_j|^2 = |x_i|^2 + |y_j|^2 - 2 (x_i . y_j)
    -> 2 cheap row-norm vectors + ONE matmul (X @ Y.T) + an outer sum.

We check it against a baremetal brute-force double loop.

Works for any N, M points in any dimension D.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. The one-matmul method
# ----------------------------------------------------------------------
def sq_dist_matmul(X, Y):
    """
    X : [N, D]  (each row a point)
    Y : [M, D]  (each row a point)
    returns Dist : [N, M],  Dist[i, j] = |X[i] - Y[j]|^2
    """
    x2 = np.sum(X * X, axis=1)          # [N]  row norms  |x_i|^2
    y2 = np.sum(Y * Y, axis=1)          # [M]  row norms  |y_j|^2
    G = X @ Y.T                         # [N, M]  THE one matmul: G[i,j] = x_i . y_j
    Dist = x2[:, None] + y2[None, :] - 2.0 * G   # outer sum + subtract
    return Dist


# ----------------------------------------------------------------------
# 2. Baremetal brute-force reference (slow, obviously correct)
# ----------------------------------------------------------------------
def sq_dist_baremetal(X, Y):
    N, D = X.shape
    M = Y.shape[0]
    Dist = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            s = 0.0
            for d in range(D):
                diff = X[i, d] - Y[j, d]
                s += diff * diff
            Dist[i, j] = s
    return Dist


# ----------------------------------------------------------------------
# 3. Check correctness
# ----------------------------------------------------------------------
def check(N, M, D, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    Y = rng.standard_normal((M, D))

    fast = sq_dist_matmul(X, Y)
    ref = sq_dist_baremetal(X, Y)

    max_err = np.max(np.abs(fast - ref))
    ok = np.allclose(fast, ref, atol=1e-9)
    print(f"N={N:4d}  M={M:4d}  D={D:3d}  ->  match={ok}   max_err={max_err:.2e}")
    return ok


if __name__ == "__main__":
    # ---- the worked 3x3 example from the notes ----
    X = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [1, 1, 0]], dtype=float)
    Y = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [2, 0, 0]], dtype=float)
    print("Worked 3x3 example:")
    print("matmul method:\n", sq_dist_matmul(X, Y))
    print("baremetal     :\n", sq_dist_baremetal(X, Y))
    print()

    # ---- random stress tests over various N, M, D ----
    print("Random correctness checks:")
    all_ok = True
    for N, M, D in [(1, 1, 1), (3, 3, 3), (5, 8, 2), (10, 4, 7), (50, 50, 3), (37, 61, 16)]:
        all_ok &= check(N, M, D)
    print()
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
