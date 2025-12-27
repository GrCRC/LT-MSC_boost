import numpy as np
import time
import sys
import gc
from sklearn.utils.extmath import randomized_svd
from solve_l1l2 import solve_l1l2


def lt_msc(X_list, gt, lambda_):
    """
    Optimized for Shared Server (1TB RAM):
    - Precision: Float32 (Memory halved, Speed 2x)
    - Math: Woodbury Matrix Identity (Avoids N*N Inversion)
    - Algo: Randomized SVD
    """

    # 1. 强制转换为 Float32
    X = [x.astype(np.float32) for x in X_list]

    V = len(X)
    N = X[0].shape[1]  # 50,000
    K = V

    print(f"[Init] Dataset: N={N}, Views={V}, Precision=Float32")
    print(f"[Init] Estimated Memory Peak: ~90 GB (Safe for shared server)")

    # 2. 初始化变量 (Float32)
    Z = [np.zeros((N, N), dtype=np.float32) for _ in range(K)]
    W = [np.zeros((N, N), dtype=np.float32) for _ in range(K)]
    G = [np.zeros((N, N), dtype=np.float32) for _ in range(K)]
    E = [np.zeros((x.shape[0], N), dtype=np.float32) for x in X]
    Y = [np.zeros((x.shape[0], N), dtype=np.float32) for x in X]

    # 预计算 XX^T (D x D)，这是 Woodbury 加速的关键
    # D (3072) << N (50000)，所以这个矩阵很小，运算极快
    XXt_list = [np.dot(x, x.T) for x in X]

    # 参数设置
    Isconverg = False
    epson = 1e-4  # 精度要求不高，1e-4 足够
    mu = 1e-4
    max_mu = 1e10
    pho_mu = 2
    rho = 1e-4
    max_rho = 1e12
    pho_rho = 2

    iter_ = 0

    while not Isconverg:
        iter_start = time.time()

        # ================= 1. Update Z (极速优化版) =================
        # 原始公式: Z = inv(I + mu/rho * X'X) * TMP
        # 优化推导: 令 A = mu/rho * X'X, H = I - TMP. 则 Z = I - inv(I+A)*H
        # Woodbury: inv(I+A) = I - X' * inv(rho/mu*I + XX') * X
        # 最终形态: Z = TMP + X' * [ inv(rho/mu*I + XX') * (X * (I - TMP)) ]

        for k in range(K):
            # 1.1 计算辅助矩阵 Q (即原公式中的 TMP 部分)
            # Q = (X'Y + mu*X'X - mu*X'E - W)/rho + G
            # 拆解: Q = (X'(Y - mu*E) - W)/rho + G + (mu/rho)X'X
            # 注意：显式算 X'X 会爆内存。我们需要避免显式计算 Q 中的 X'X 项。

            # 换一种更直接的 Woodbury 形式:
            # 目标方程: (rho*I + mu*X'X) Z = X'(Y + mu*X - mu*E) + rho*G - W
            # 令 RHS = X'(Y + mu*X - mu*E) + rho*G - W
            # Z = inv(rho*I + mu*X'X) * RHS
            # Z = 1/rho * [ RHS - X' * inv(rho/mu*I + XX') * (X * RHS) ]

            # Step A: 计算 RHS (Right Hand Side)
            # M = Y + mu*X - mu*E  (Size: D x N)
            M = Y[k] + mu * X[k] - mu * E[k]

            # RHS = X.T @ M + rho*G - W
            # X.T @ M (N x N) 是本算法中最耗时的步骤之一
            RHS = np.dot(X[k].T, M)
            RHS += rho * G[k]
            RHS -= W[k]

            # Step B: Woodbury 核心小矩阵求逆 (D x D)
            alpha = rho / mu
            # matrix_to_inv = alpha * I + XX^T
            matrix_to_inv = XXt_list[k].copy()
            flat_idx = np.arange(X[k].shape[0])
            matrix_to_inv[flat_idx, flat_idx] += alpha

            inv_P = np.linalg.inv(matrix_to_inv)  # D x D, 瞬间完成

            # Step C: 链式乘法 (避免 N*N 中间变量)
            # Term = X' * (inv_P * (X * RHS))

            # 1. X @ RHS -> (D x N) * (N x N) -> (D x N) [耗时，但必须]
            X_RHS = np.dot(X[k], RHS)

            # 2. inv_P @ X_RHS -> (D x D) * (D x N) -> (D x N) [快]
            temp = np.dot(inv_P, X_RHS)

            # 3. X.T @ temp -> (N x D) * (D x N) -> (N x N) [耗时]
            correction = np.dot(X[k].T, temp)

            # Final Z
            Z[k] = (RHS - correction) / rho

            # 释放大矩阵内存
            del RHS, M, X_RHS, temp, correction

        # ================= 2. Update E =================
        F_list = []
        for v in range(K):
            # XZ = X @ Z
            XZ = np.dot(X[v], Z[v])
            F_k = X[v] - XZ + Y[v] / mu
            F_list.append(F_k)
            del XZ

        F_concat = np.vstack(F_list)
        E_concat = solve_l1l2(F_concat, lambda_ / mu)

        beg_ind = 0
        for v in range(K):
            end_ind = beg_ind + X[v].shape[0]
            E[v] = E_concat[beg_ind:end_ind, :]
            beg_ind = end_ind

        del F_concat, E_concat, F_list

        # ================= 3. Update Y =================
        for k in range(K):
            # Y = Y + mu * (X - XZ - E)
            XZ = np.dot(X[k], Z[k])
            Y[k] += mu * (X[k] - XZ - E[k])
            del XZ

        # ================= 4. Update G (Randomized SVD) =================
        for k in range(K):
            Target = Z[k] + W[k] / rho

            # 使用 Randomized SVD 极速分解
            # n_components=600: 既然只要小数点后4位精度，保留前600个奇异值足够了
            # 这比全量 SVD 快几百倍
            U, S_vals, Vt = randomized_svd(
                Target,
                n_components=600,
                n_iter=5,  # 迭代次数少一点以提速
                random_state=42
            )

            # 软阈值
            thresh = 1.0 / rho
            S_vals = np.maximum(0, S_vals - thresh)

            # 重构 G
            G[k] = (U * S_vals) @ Vt

            del Target, U, S_vals, Vt

        # ================= 5. Update W =================
        for k in range(K):
            W[k] += rho * (Z[k] - G[k])

        # ---------------- 收敛检查 (抽样加速) ----------------
        # 全量计算范数太慢，随机抽样 5000 个点检查误差
        max_res = 0
        idx = np.random.randint(0, N, 5000)
        for k in range(K):
            res1 = np.max(np.abs(Z[k][idx, :] - G[k][idx, :]))
            max_res = max(max_res, res1)

        print(f"Iter {iter_ + 1}: Max Res (Approx) = {max_res:.6f}, Time = {time.time() - iter_start:.2f}s")
        sys.stdout.flush()

        if max_res < epson and iter_ > 20:  # 至少迭代20次
            Isconverg = True

        if iter_ >= 30:  # 最大迭代限制
            Isconverg = True

        iter_ += 1
        mu = min(mu * pho_mu, max_mu)
        rho = min(rho * pho_rho, max_rho)
        gc.collect()  # 显式垃圾回收，防止内存泄露

    # ---------------- 最终构建图 ----------------
    print("Optimization Done. Building Affinity Matrix...")
    S = np.zeros((N, N), dtype=np.float32)
    for k in range(K):
        S += np.abs(Z[k]) + np.abs(Z[k].T)

    # 转换为 float16 进一步压缩存储 (可选，但 float32 更通用)
    return S