import numpy as np
import scipy.io as sio
import time
import os
import sys
import gc
from scipy.linalg import solve
from sklearn.utils.extmath import randomized_svd

import psutil
import os


# === 新增内存监控辅助函数 ===
def print_memory_usage(tag=""):
    """
    打印当前进程占用的物理内存 (RSS)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # 将字节转换为 GB
    mem_gb = mem_info.rss / 1024 / 1024 / 1024
    print(f"[{tag}] Current Memory: {mem_gb:.2f} GB")

# ==========================================
# 核心算法实现部分 (优化版)
# ==========================================

def solve_l1l2(W, lambda_val):
    """
    求解 L1/L2 范数最小化问题: min lambda |E|_2,1 + |E - W|_F^2
    """
    # 确保输入是 float32
    W = W.astype(np.float32)
    E = np.zeros_like(W, dtype=np.float32)

    # 计算每一列的 L2 范数
    wnorms = np.linalg.norm(W, axis=0)

    # 软阈值操作
    scale = np.maximum(0, wnorms - lambda_val) / (wnorms + 1e-10)

    # 将 scale 广播到每一列
    E = W * scale[np.newaxis, :]
    return E


def softth_randomized(F, lambda_val, n_components=300):
    """
    使用随机化 SVD 进行加速的软阈值操作
    """
    if F.shape[0] > 2000 or F.shape[1] > 2000:
        # 大矩阵使用 Randomized SVD
        U, S, Vt = randomized_svd(F, n_components=n_components, random_state=42)
    else:
        # 小矩阵使用普通 SVD
        U, S, Vt = np.linalg.svd(F, full_matrices=False)

    S_thresh = np.maximum(0, S - lambda_val)

    # 如果所有奇异值都被截断，直接返回零矩阵
    idx = S_thresh > 0
    if np.sum(idx) == 0:
        return np.zeros_like(F)

    U = U[:, idx]
    S_thresh = S_thresh[idx]
    Vt = Vt[idx, :]

    # 重构矩阵: (U * S) @ Vt，注意顺序以优化速度
    E = (U * S_thresh) @ Vt
    return E.astype(np.float32)


def update_tensor_variables(WT, Z_tensor, sX, mu, rho):
    """
    更新张量变量 G 和 W
    对应原始代码中的 updateG_tensor 和 update W 部分
    """
    N, _, K = sX
    Target = Z_tensor + WT / rho

    # 初始化 G_new
    G_new = np.zeros(sX, dtype=np.float32)

    # 多视图聚类通常关注视图间的一致性 (Mode-3) 和样本自表达 (Mode-1/2)
    # 这里我们对 Mode-3 (视图维) 进行低秩约束，这是多视图聚类的核心
    # 将张量沿第3维展开 -> (K, N*N)

    # Mode-3 Unfolding
    # Shape: (K, N*N)
    mat_mode3 = np.reshape(np.moveaxis(Target, 2, 0), (K, N * N), order='F')

    # 对 (K, N*N) 矩阵做软阈值。因为 K 很小 (如3)，这里SVD极快
    mat_th = softth_randomized(mat_mode3, 1.0 / rho, n_components=K)

    # Fold back to Tensor
    G_new = np.moveaxis(np.reshape(mat_th, (K, N, N), order='F'), 0, 2)

    # 如果内存允许，也可以加入 Mode-1 的约束（对应样本相关性），但对内存要求极高
    # 下面注释掉的代码是 Mode-1 的实现，如果内存不足 128G 建议不要开启
    """
    # Mode-1 Unfolding: (N, N*K) -> Huge Matrix!
    mat_mode1 = np.reshape(np.moveaxis(Target, 0, 0), (N, N*K), order='F')
    mat_th1 = softth_randomized(mat_mode1, 1.0/rho, n_components=300)
    G_mode1 = np.moveaxis(np.reshape(mat_th1, (N, N, K), order='F'), 0, 0)
    G_new = (G_new + G_mode1) / 2.0
    """

    # 更新对偶变量 W
    # W = W + rho * (Z - G)
    W_new = WT + rho * (Z_tensor - G_new)

    return G_new, W_new


def lt_msc(X_list, gt, lambda_val):
    """
    LT-MSC 主函数 (优化内存版)
    """
    # 1. 强制转换为 Float32
    X = [x.astype(np.float32) for x in X_list]
    print_memory_usage("Start")

    V_num = len(X)
    N = X[0].shape[1]
    D_list = [x.shape[0] for x in X]

    print(f"[Init] Dataset: N={N}, Views={V_num}, Dims={D_list}")
    print(f"[Init] Estimated Z-Tensor Size: {N * N * V_num * 4 / 1024 ** 3:.2f} GB")

    # 2. 初始化变量 (Float32)
    Z_tensor = np.zeros((N, N, V_num), dtype=np.float32)
    G_tensor = np.zeros((N, N, V_num), dtype=np.float32)
    W_tensor = np.zeros((N, N, V_num), dtype=np.float32)

    E = [np.zeros((d, N), dtype=np.float32) for d in D_list]
    Y = [np.zeros((d, N), dtype=np.float32) for d in D_list]

    print_memory_usage("After Init")

    # 3. 预计算 Woodbury 矩阵 XX^T (D x D)，避免计算 N x N
    XXt_list = []
    for v in range(V_num):
        XXt_list.append(X[v] @ X[v].T)

    # 参数设置
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

        # --- 1. Update Z (使用 Woodbury 恒等式) ---
        for k in range(V_num):
            # 目标方程: (alpha * I + X'X) Z = RHS
            # RHS = X'(Y/mu + X - E) + alpha*G - alpha*W/mu

            # Step A: 构建 RHS (Right Hand Side)
            M = Y[k] / mu + X[k] - E[k]

            # 注意：这里 X[k].T @ M 会生成一个 N x N 的稠密矩阵
            # 这是内存消耗的峰值点之一。由于 Z_tensor 本身就是 N x N，这是不可避免的。
            RHS = np.dot(X[k].T, M)
            del M  # 及时释放

            alpha = rho / mu
            RHS += alpha * G_tensor[:, :, k]
            RHS -= (1.0 / mu) * W_tensor[:, :, k]

            # Step B: Woodbury 求逆核心 (D x D)
            # inv(alpha*I + X'X) = 1/alpha * (I - X' * inv(alpha*I + XX') * X)

            # Core inverse: (alpha*I + XX')
            woodbury_core = XXt_list[k] + alpha * np.eye(D_list[k], dtype=np.float32)

            # Step C: 计算修正项
            # temp = inv(core) * (X * RHS)
            # X * RHS -> (D x N)
            X_RHS = np.dot(X[k], RHS)

            # Solve linear system instead of explicit inverse
            temp_D_N = solve(woodbury_core, X_RHS, assume_a='pos')
            del X_RHS

            # correction = X' * temp
            correction = np.dot(X[k].T, temp_D_N)
            del temp_D_N

            # Final Z
            Z_new = (RHS - correction) / alpha
            del RHS, correction

            # 更新 Z_tensor
            Z_tensor[:, :, k] = Z_new
            del Z_new

            # 强制垃圾回收
            gc.collect()

        # 在 Z 更新完后查看内存，确认是否释放回落
        print_memory_usage(f"Iter {iter_idx} - After Z Update")

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

        # --- 4. Update G and W (Tensor Constraints) ---
        G_new, W_new = update_tensor_variables(W_tensor, Z_tensor, (N, N, V_num), mu, rho)

        # 检查收敛性 (采样近似，避免全量计算)
        idx = np.random.randint(0, N, 2000)
        diff_g = np.max(np.abs(G_tensor[idx, :, :] - G_new[idx, :, :]))
        max_diff = max(max_diff, diff_g)

        G_tensor = G_new
        W_tensor = W_new
        del G_new, W_new

        print_memory_usage(f"Iter {iter_idx} - End")  # <--- 监控点 3：迭代结束

        # --- Update Params ---
        iter_idx += 1
        mu = min(mu * pho_mu, max_mu)
        rho = min(rho * pho_rho, max_rho)

        print(f"Iter {iter_idx}: Max Diff (Approx) = {max_diff:.6f}, Time = {time.time() - iter_start:.2f}s")
        sys.stdout.flush()

        if max_diff < epsilon and iter_idx > 15:
            is_converged = True

    # --- 构建相似度矩阵 ---
    print("Optimization Done. Building Affinity Matrix...")
    S = np.zeros((N, N), dtype=np.float32)
    for k in range(V_num):
        S += np.abs(Z_tensor[:, :, k]) + np.abs(Z_tensor[:, :, k].T)

    return S


# ==========================================
# 数据加载与主程序 (复刻 ye.py)
# ==========================================

if __name__ == "__main__":
    # 添加当前目录到路径
    sys.path.append(os.getcwd())

    # --------------------------------------------------------
    # 配置部分：根据 ye.py 的逻辑设置数据集
    # --------------------------------------------------------
    # 你可以修改这里的路径指向你的实际文件
    # 示例假设使用 cifar100 (结构类似 cifar10)
    dataName = "Cifar100"
    file_path = '../data/cifar100.mat'  # 请修改为你的真实路径，例如 'cifar10.mat'

    print(f"Loading data from {file_path}...")

    try:
        mat_dict = sio.loadmat(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please check the file path.")
        sys.exit(1)

    # 按照 ye.py 的读取逻辑
    # 假设 .mat 文件结构包含 'data' 和 'truelabel'
    # 注意：根据 ye.py，raw_data 是一个 cell array (object array in numpy)
    if 'data' in mat_dict:
        raw_data = mat_dict['data']
        # 解析数据: ye.py 中使用了 raw_data[0,0], raw_data[1,0]...
        # 这通常意味着 MATLAB 中的 data{1}, data{2}...
        # 我们这里动态读取所有视图

        # 检查 raw_data 的形状
        if raw_data.shape[0] > raw_data.shape[1]:
            # 如果是 (V, 1) 的形状
            num_views_data = raw_data.shape[0]
            x = [raw_data[v, 0] for v in range(num_views_data)]
        else:
            # 如果是 (1, V) 的形状
            num_views_data = raw_data.shape[1]
            x = [raw_data[0, v] for v in range(num_views_data)]

        # 标签处理
        if 'truelabel' in mat_dict:
            y = mat_dict['truelabel'].flatten()
        elif 'gt' in mat_dict:
            y = mat_dict['gt'].flatten()
        else:
            # 尝试寻找其他可能的标签键名
            print("Warning: Label key not found (truelabel/gt). Using dummy labels.")
            y = np.zeros(x[0].shape[1])

    else:
        # 兼容其他格式 (如 Hdigit / scene)
        # 尝试直接查找 X1, X2...
        x = []
        i = 1
        while f'X{i}' in mat_dict or f'x{i}' in mat_dict:
            key = f'X{i}' if f'X{i}' in mat_dict else f'x{i}'
            data_view = mat_dict[key]
            # 检查是否需要转置：我们期望 (D, N)
            # 这是一个启发式检查，通常样本数 N 比较大
            if data_view.shape[0] > data_view.shape[1]:
                # 如果是 (N, D)，转置为 (D, N)
                data_view = data_view.T
            x.append(data_view)
            i += 1

        if 'Y' in mat_dict:
            y = mat_dict['Y'].flatten()
        elif 'gt' in mat_dict:
            y = mat_dict['gt'].flatten()
        else:
            y = np.zeros(x[0].shape[1])

    # --------------------------------------------------------
    # 预处理部分 (复刻 ye.py)
    # --------------------------------------------------------

    # 1. 强制转换为 float32 并归一化
    x = [view.astype(np.float32) for view in x]

    gt = y.flatten()
    # gt = gt + 1 # Python 索引从 0 开始，通常不需要加 1，除非后续评估代码有要求

    num_views = len(x)
    print(f"Number of views: {num_views}")
    print(f"Sample size (N): {x[0].shape[1]}")

    # 归一化 (L2 Norm per column)
    print("Normalizing data...")
    for v in range(num_views):
        # 每一列除以其L2范数
        norms = np.sqrt(np.sum(x[v] ** 2, axis=0)) + 1e-10
        x[v] = x[v] / norms

    # --------------------------------------------------------
    # 执行 LT-MSC
    # --------------------------------------------------------
    t1 = time.time()

    lambda_val = 0.1  # 正则化参数

    print(f"Starting LT-MSC with lambda={lambda_val}...")
    try:
        S = lt_msc(x, gt, lambda_val)
    except MemoryError:
        print("\n[CRITICAL ERROR] Memory Limit Exceeded.")
        print("Suggestion: Try reducing the dataset size or running on a machine with >128GB RAM.")
        sys.exit(1)

    t2 = time.time() - t1
    print(f"Total Time elapsed: {t2:.2f} seconds")

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------
    save_path = f'W_{dataName}.npz'
    print("Saving graph to {}".format(save_path))
    np.savez(save_path, S=S, gt=gt)
    print("Save complete.")

    # 如果有对应的聚类评估代码 (spectral clustering)，可以在此处调用
    # 例如:
    # from sklearn.cluster import SpectralClustering
    # from sklearn.metrics import normalized_mutual_info_score, accuracy_score
    # n_clusters = len(np.unique(gt))
    # sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    # labels = sc.fit_predict(S)
    # print(f"NMI: {normalized_mutual_info_score(gt, labels)}")