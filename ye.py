# import numpy as np
# import time
# import scipy.io
#
#
# data = load('../data/BDGP.mat')
# x{1}=X1
# x{2}=im2double(X2')
# y=Y
#
#
# # gt = y
# gt = y.copy()
# gt = gt + 1  # MATLAB 索引从1开始，Python从0开始，此处保持一致性
#
# num_views = len(x)  # 得到视图数量
#
# # 归一化
# for v in range(num_views):
#     # 每一列除以其L2范数，避免除零
#     norm = np.sqrt(np.sum(x[v] ** 2, axis=0)) + 1e-10
#     x[v] = x[v] / norm
#
# # 计时
# t1 = time.process_time()
#
# lambda_ = 0.1
# acc, nmi, f, ri, ar = lt_msc(x, gt, lambda_)
#
# t2 = time.process_time() - t1
#
# print(f"acc={acc:.4f}, nmi={nmi:.4f}, f={f:.4f}, ri={ri:.4f}, ar={ar:.4f}")
# print(f"Time elapsed: {t2:.4f} seconds")


import numpy as np
import scipy.io as sio
import time
import os
import sys
from lt_msc import lt_msc

# 添加当前目录到路径（模拟MATLAB的addpath）
sys.path.append(os.getcwd())

# #加载数据
# data1 = sio.loadmat('../../data/scene1.mat')
# x1 = data1['X1'].T  # 转置，相当于MATLAB中的'
# data2 = sio.loadmat('../../data/scene2.mat')
# x2 = data2['X2'].T
# data3 = sio.loadmat('../../data/scene3.mat')
# x3 = data3['X3'].T
# data4 = sio.loadmat('../../data/scene4.mat')
# x4 = data4['X4'].T
# data_label = sio.loadmat('../../data/scenelabel.mat')
# y = data_label['gt']
# x = [x1, x2, x3, x4]

# dataName = "Hdigit"
# data1 = sio.loadmat('../data/Hdigit1.mat')
# x1 = data1['X1'].T  # 转置，相当于MATLAB中的'
# data2 = sio.loadmat('../data/Hdigit2.mat')
# x2 = data2['X2'].T
# data_label = sio.loadmat('../data/Hdigit_lable.mat')
# y = data_label['gt']
# x = [x1, x2]

dataName = "Cifar100"
file_path = '../data/cifar100.mat'
mat_dict = sio.loadmat(file_path)

raw_data = mat_dict['data']
x1 = raw_data[0,0]
x2 = raw_data[1,0]
x3 = raw_data[2,0]
x = [x1, x2, x3]
y = mat_dict['truelabel'].flatten()
x = [view.astype(float) for view in x]

# # 加载数据
# data = sio.loadmat('../../data/wine.mat')
#
# # 获取wine_data和wine_id
# wine_data = data['wine_data']
# wine_id = data['wine_id'].flatten()  # 将wine_id转换为一维数组
#
# # 创建视图列表，对每个视图进行转置（相当于MATLAB中的'）
# x = [
#     wine_data[0, 0].T,  # wine_data{1,1}'
#     wine_data[0, 1].T,  # wine_data{1,2}'
#     wine_data[0, 2].T   # wine_data{1,3}'
# ]
#
# y = wine_id


# 创建视图列表

gt = y.flatten()  # 将gt转换为一维数组
gt = gt + 1  # 相当于MATLAB中的gt=gt+1

num_views = len(x)  # 得到视图数量
print(f"num_views = {num_views}")

# 归一化
for v in range(num_views):
    # 计算每列的L2范数，加上小常数防止除零
    norms = np.sqrt(np.sum(x[v]**2, axis=0)) + 1e-10
    x[v] = x[v] / norms

# 记录开始时间
t1 = time.time()

lambda_val = 0.1

# 注意：这里需要实现lt_msc函数
# 由于原MATLAB代码中没有提供lt_msc的实现，这里假设它是一个函数
# 你需要根据实际的lt_msc算法来实现这个函数
S = lt_msc(x, gt, lambda_val)

t2 = time.time() - t1
print(f"Total Time elapsed: {t2:.2f} seconds")

save_path = f'../Graph/W_{dataName}.npz'
print("Saving graph to {}".format(save_path))

np.savez(save_path, S = S)
print("Save complete")