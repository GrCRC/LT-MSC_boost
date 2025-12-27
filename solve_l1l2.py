import numpy as np

def solve_l2(w, lambda_):
    """
    Solve min_x lambda * ||x||_2 + ||x - w||_2^2
    => closed form: x = (1 - lambda / ||w||_2)_+ * w
    """
    nw = np.linalg.norm(w)
    if nw > lambda_:
        x = (1 - lambda_ / nw) * w
    else:
        x = np.zeros_like(w)
    return x


def solve_l1l2(W, lambda_):
    """
    Column-wise L1/L2 minimization:
    For each column w_i in W, solve:
        min_x lambda * ||x||_2 + ||x - w_i||_2^2
    """
    n = W.shape[1]
    E = np.zeros_like(W)
    for i in range(n):
        E[:, i] = solve_l2(W[:, i], lambda_)
    return E
