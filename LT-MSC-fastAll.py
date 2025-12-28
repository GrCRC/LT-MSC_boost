import numpy as np
import scipy.io as sio
import time
import os
import sys
import gc
import psutil
from scipy.linalg import solve
from sklearn.utils.extmath import randomized_svd


# ==========================================
# 内存监控工具
# ==========================================
def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024 / 1024 / 1024
    print(f"[{tag}] Current Memory: {mem_gb:.2f} GB")
    sys.stdout.flush()


# ==========================================
# 核心算法实现部分 (极大内存优化版)
# ==========================================

def solve_l1l2(W, lambda_val):
    """
    求解 L1/L2 范数最小化问题
    """
    W = W.astype(np.float32)
    # W is (D, N), D is small (e.g. 2048), N is 50000.
    # This takes ~400MB. Safe to compute directly.
    wnorms = np.linalg.norm(W, axis=0)
    scale = np.maximum(0, wnorms - lambda_val) / (wnorms + 1e-10)
    E = W * scale[np.newaxis, :]
    return E


def compute_shrinkage_operator(Z_tensor, W_tensor, rho, N, K, chunk_size=5000):
    """
    计算 Soft Thresholding 的核心收缩算子 T (3x3矩阵)。
    避免显式构造巨大的 (N*N, K) 矩阵。
    """
    # 逻辑：
    # 设 M_tall 为 (N^2, K) 的矩阵，其中每一行对应 tensor 在 (i,j) 位置的 fiber: z + w/rho
    # 我们需要对 M_tall^T (K, N^2) 做 Soft Thresholding。
    # 设 M_tall^T = U S V^T (这里 U 是 KxK, S 是 K, V 是 N^2 x K)
    # 阈值操作后: G_tall^T = U * thresh(S) * V^T
    # 变换为右乘形式: G_tall = M_tall * T
    # 其中 T = V S^{-1} thresh(S) V^T (不对，推导见下)

    # 正确推导:
    # M_tall = Q Sigma P^T (SVD of M_tall, Q is N^2xK, P is KxK)
    # M_tall^T = P Sigma Q^T.
    # SoftTh(M_tall^T) = P thresh(Sigma) Q^T.
    # Target G_tall = (P thresh(Sigma) Q^T)^T = Q thresh(Sigma) P^T.
    # 我们有 M_tall = Q Sigma P^T => Q = M_tall P Sigma^{-1}
    # 代入: G_tall = (M_tall P Sigma^{-1}) thresh(Sigma) P^T
    #              = M_tall * [P * diag(thresh(sigma)/sigma) * P^T]
    # 令 T = P * diag(thresh(sigma)/sigma) * P^T
    # 这是一个 K x K 的矩阵。

    # 1. 计算 M_tall^T @ M_tall (即协方差矩阵，K x K)
    #    M_tall = Z_flat + W_flat/rho
    MMt = np.zeros((K, K), dtype=np.float32)

    # 使用分块累加，避免分配 N^2 x K 的大矩阵
    total_elements = N * N

    # Z_tensor 在内存中是 (N, N, K)，展平最后两维比较麻烦，但我们只需要视为 (N*N, K)
    # 由于 K 在最后，reshape(-1, K) 对于 C-order 是零拷贝视图！
    Z_flat = Z_tensor.reshape(-1, K)
    W_flat = W_tensor.reshape(-1, K)

    for i in range(0, total_elements, chunk_size):
        end = min(i + chunk_size, total_elements)
        # 获取切片 (Chunk, K)
        z_chunk = Z_flat[i:end, :]
        w_chunk = W_flat[i:end, :]

        # m_chunk = z + w/rho
        m_chunk = z_chunk + w_chunk * (1.0 / rho)

        # 累加 KxK
        MMt += np.dot(m_chunk.T, m_chunk)

        del m_chunk, z_chunk, w_chunk

    # 2. 对 KxK 矩阵做特征分解 (Eigendecomposition)
    # MMt = P Sigma^2 P^T
    evals, P = np.linalg.eigh(MMt)

    # 特征值可能有些微负值（数值误差），修正为0
    evals = np.maximum(0, evals)
    sigmas = np.sqrt(evals)

    # 3. 计算收缩系数
    # lambda = 1/rho
    thresh_val = 1.0 / rho
    sigmas_thresholded = np.maximum(0, sigmas - thresh_val)

    # 处理除零 (如果 sigma=0, scale=0)
    scale = np.zeros_like(sigmas)
    nonzero_idx = sigmas > 1e-10
    scale[nonzero_idx] = sigmas_thresholded[nonzero_idx] / sigmas[nonzero_idx]

    # 4. 构造 T 矩阵 (K x K)
    # T = P * diag(scale) * P^T
    T = np.dot(P * scale, P.T)

    return T.astype(np.float32)


def update_tensor_blockwise(Z_tensor, W_tensor, G_tensor, rho, T, chunk_size=100000):
    """
    分块更新 G 和 W，完全避免大内存分配。
    G = (Z + W/rho) * T
    W = W + rho * (Z - G)
    """
    N = Z_tensor.shape[0]
    K = Z_tensor.shape[2]
    total_elements = N * N

    # 获取视图 (View)
    Z_flat = Z_tensor.reshape(-1, K)
    W_flat = W_tensor.reshape(-1, K)
    G_flat = G_tensor.reshape(-1, K)  # 直接修改 G_tensor 的内存

    # 分块处理
    for i in range(0, total_elements, chunk_size):
        end = min(i + chunk_size, total_elements)

        # 1. 计算 Target Chunk
        # target = z + w/rho (分配小内存)
        target_chunk = Z_flat[i:end, :] + W_flat[i:end, :] * (1.0 / rho)

        # 2. 更新 G Chunk
        # G = Target * T
        # 这里的 G_flat[...] 写入直接作用于 G_tensor 原地
        g_chunk = np.dot(target_chunk, T)
        G_flat[i:end, :] = g_chunk

        # 3. 更新 W Chunk
        # W = W + rho * (Z - G)
        # 直接原地修改 W_tensor
        # W_flat[i:end, :] += rho * (Z_flat[i:end, :] - g_chunk)
        # 为了更省内存，逐步计算
        diff = Z_flat[i:end, :] - g_chunk
        W_flat[i:end, :] += diff * rho

        del target_chunk, g_chunk, diff

    # 强制垃圾回收
    gc.collect()


def lt_msc(X_list, gt, lambda_val):
    # 1. 基础配置
    X = [x.astype(np.float32) for x in X_list]
    V_num = len(X)
    N = X[0].shape[1]
    D_list = [x.shape[0] for x in X]

    print(f"[Init] Dataset: N={N}, Views={V_num}, Dims={D_list}")
    print_memory_usage("Start")

    # 2. 内存分配 (Peak Memory Point 1)
    # 此时约占用 9.3GB * 3 * 3 (Z,G,W) = 84GB
    # 加上数据本身，已经接近 90GB，但这是静态的，不会增长。
    print("Allocating Tensors...")
    Z_tensor = np.zeros((N, N, V_num), dtype=np.float32)
    G_tensor = np.zeros((N, N, V_num), dtype=np.float32)
    W_tensor = np.zeros((N, N, V_num), dtype=np.float32)

    E = [np.zeros((d, N), dtype=np.float32) for d in D_list]
    Y = [np.zeros((d, N), dtype=np.float32) for d in D_list]

    print_memory_usage("After Alloc")

    # 3. Woodbury 预计算
    XXt_list = []
    for v in range(V_num):
        XXt_list.append(X[v] @ X[v].T)

    # 参数
    max_iter = 50
    rho = 1e-4
    max_rho = 1e10
    pho_rho = 2
    mu = 1e-4
    max_mu = 1e10
    pho_mu = 2
    epsilon = 1e-4

    iter_idx = 0
    is_converged = False

    print("Optimization Start...")

    while not is_converged and iter_idx < max_iter:
        iter_start = time.time()
        max_diff = 0

        # --- 1. Update Z (Woodbury) ---
        for k in range(V_num):
            M = Y[k] / mu + X[k] - E[k]
            # 这里会生成一个 9.3GB 的 RHS 矩阵，是瞬时内存峰值
            # 必须确保在循环末尾释放
            RHS = np.dot(X[k].T, M)
            del M

            alpha = rho / mu
            RHS += alpha * G_tensor[:, :, k]
            RHS -= (1.0 / mu) * W_tensor[:, :, k]

            # Woodbury Inverse
            woodbury_core = XXt_list[k] + alpha * np.eye(D_list[k], dtype=np.float32)
            X_RHS = np.dot(X[k], RHS)  # (D, N)

            temp_D_N = solve(woodbury_core, X_RHS, assume_a='pos')
            del X_RHS

            correction = np.dot(X[k].T, temp_D_N)  # (N, N) - 9.3GB
            del temp_D_N

            # Update Z slice in-place
            Z_tensor[:, :, k] = (RHS - correction) / alpha

            del RHS, correction
            gc.collect()  # 关键：释放 RHS 和 correction 占用的 ~18GB

        # print_memory_usage(f"Iter {iter_idx} - After Z")

        # --- 2. Update E ---
        for k in range(V_num):
            XZ = np.dot(X[k], Z_tensor[:, :, k])
            F = X[k] - XZ + Y[k] / mu
            E[k] = solve_l1l2(F, lambda_val / mu)
            del XZ, F

        # --- 3. Update Y ---
        for k in range(V_num):
            XZ = np.dot(X[k], Z_tensor[:, :, k])
            Y[k] += mu * (X[k] - XZ - E[k])
            del XZ

        # --- 4. Update G and W (Block-wise Streaming) ---
        # 这一步彻底重构，不再分配大内存

        # 4.1 计算收缩算子 T (K x K)
        T_matrix = compute_shrinkage_operator(Z_tensor, W_tensor, rho, N, V_num)

        # 4.2 记录 G 的旧值用于收敛检查 (只采样，不全量拷贝)
        idx_sample = np.random.randint(0, N, 2000)
        G_old_sample = G_tensor[idx_sample, :, :].copy()

        # 4.3 流式更新 G 和 W
        update_tensor_blockwise(Z_tensor, W_tensor, G_tensor, rho, T_matrix)

        # 检查收敛性 (Approx)
        G_new_sample = G_tensor[idx_sample, :, :]
        diff_g = np.max(np.abs(G_old_sample - G_new_sample))
        max_diff = max(max_diff, diff_g)
        del G_old_sample, G_new_sample

        print_memory_usage(f"Iter {iter_idx} - End")

        # --- Update Params ---
        iter_idx += 1
        mu = min(mu * pho_mu, max_mu)
        rho = min(rho * pho_rho, max_rho)

        print(f"Iter {iter_idx}: Max Diff = {max_diff:.6f}, Time = {time.time() - iter_start:.2f}s")
        sys.stdout.flush()

        if max_diff < epsilon and iter_idx > 15:
            is_converged = True

    print("Building Affinity Matrix...")
    S = np.zeros((N, N), dtype=np.float32)
    for k in range(V_num):
        S += np.abs(Z_tensor[:, :, k]) + np.abs(Z_tensor[:, :, k].T)

    return S


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # 模拟/加载数据
    # 请确保路径正确
    file_path = '../data/cifar100.mat'
    dataName = "Cifar100"

    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print("File not found. Generating Dummy Data for test...")
        # 生成假数据测试内存
        N_test = 50000
        V_test = 3
        X = [np.random.randn(512, N_test).astype(np.float32) for _ in range(V_test)]
        gt = np.random.randint(0, 10, N_test)
    else:
        mat_dict = sio.loadmat(file_path)
        if 'data' in mat_dict:
            raw_data = mat_dict['data']
            if raw_data.shape[0] > raw_data.shape[1]:
                x = [raw_data[v, 0] for v in range(raw_data.shape[0])]
            else:
                x = [raw_data[0, v] for v in range(raw_data.shape[1])]
            y = mat_dict['truelabel'].flatten() if 'truelabel' in mat_dict else mat_dict['gt'].flatten()
        else:
            # 简单的 Fallback
            x = []
            i = 1
            while f'X{i}' in mat_dict or f'x{i}' in mat_dict:
                k = f'X{i}' if f'X{i}' in mat_dict else f'x{i}'
                d = mat_dict[k]
                if d.shape[0] > d.shape[1]: d = d.T
                x.append(d)
                i += 1
            y = mat_dict['gt'].flatten() if 'gt' in mat_dict else np.zeros(x[0].shape[1])

        X = x
        gt = y

    # 归一化
    for v in range(len(X)):
        norms = np.sqrt(np.sum(X[v] ** 2, axis=0)) + 1e-10
        X[v] = X[v] / norms

    try:
        S = lt_msc(X, gt, lambda_val=0.1)
        np.savez(f'W_{dataName}.npz', S=S, gt=gt)
        print("Done.")
    except MemoryError:
        print("OOM triggered.")