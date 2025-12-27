import numpy as np
from sklearn.utils.extmath import randomized_svd


def updateG_tensor(WT, K, sX, mu, gamma, V, mode):
    # 将 WT 拆分为每个视图的 W
    W = [WT[:, :, v] for v in range(V)]

    # 张量展开成向量
    w = Tensor2Vector(W, sX[0], sX[1], sX[2], V)
    k = Tensor2Vector(K, sX[0], sX[1], sX[2], V)

    wk = k + w / mu
    WKten = Vector2Tensor(wk, sX[0], sX[1], sX[2], V)

    WK = Tensor2Matrix(WKten, mode, sX[0], sX[1], sX[2])
    WK1 = softth(WK, gamma[mode - 1] / mu)  # MATLAB 的索引从1开始，Python从0开始
    my_tensor = WK1.reshape(sX)

    return my_tensor


# ======== 辅助函数部分 ========

def Vector2Tensor(v, dim1, dim2, dim3, K):
    L = v.shape[0] // K
    T = []
    for k in range(K):
        start = k * L
        end = (k + 1) * L
        T_k = v[start:end].reshape((dim1, dim2))
        T.append(T_k)
    return T


def Tensor2Vector(T, dim1, dim2, dim3, K):
    v_list = []
    for k in range(K):
        v_list.append(T[k].reshape(dim1 * dim2, 1))
    return np.vstack(v_list)


def Tensor2Matrix(T, unfolding_mode, dim1, dim2, dim3):
    if unfolding_mode == 1:
        # 沿 mode-1 展开
        m = np.hstack(T)
    elif unfolding_mode == 2:
        # 沿 mode-2 展开
        m = np.vstack([t.T for t in T])
    elif unfolding_mode == 3:
        # 沿 mode-3 展开
        m = np.hstack([t.T.reshape(-1, 1) for t in T])
    else:
        raise ValueError("Invalid unfolding_mode. Must be 1, 2, or 3.")
    return m

#
# def softth(F, lam):
#     # SVD奇异值软阈值化
#     U, S, Vt = np.linalg.svd(F, full_matrices=False)
#     diagS = np.maximum(0, S - lam)
#     svp = np.sum(diagS > 0)
#     if svp < 0.5:
#         svp = 1
#     E = U[:, :svp] @ np.diag(diagS[:svp]) @ Vt[:svp, :]
#     return E

def softth(F, lam):
    U, S, Vt = randomized_svd(F,n_components=600, random_state=42)
    diagS = np.maximum(0, S - lam)
    svp = np.sum(diagS > 0)

    if svp < 0.5:
        svp = 1

    E = U[:, : svp] @ np.diag(diagS[:svp]) @ U[:svp,:]

    return E