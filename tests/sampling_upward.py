# 由下及上的多进程辅助抽样


from mpi4py import MPI
import numpy as np
import random
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# 随机数种子，留空默认使用当前时间进行设置
np.random.seed(rank)
random.seed(rank)

n = 5
m = int(np.log(size) / np.log(2))
if 2**n % size != 0:
    if rank == 0:
        print('Improper No. of procs')    

# 随机一个用于测试的量子态
# 只有 0 号进程上的量子态才会用于后续计算，其他进程上的量子态被摒弃
state = np.random.random(2**n)
state = state / np.sqrt(sum(state * state))
state_part = np.zeros(2**(n-m), float)

# 将 0 号进程上的量子态向量均匀切分给通信子内的所有进程
comm.Scatterv(state, state_part)
prob_part = state_part * state_part.conj()

# 每个进程持有的片段的被抽中几率
S = prob_part.sum()
# prob_part = prob_part / S

# 每个进程实际分担的抽样次数
alpha = 1.001
N = 1000
N_part = int(N * S * alpha)

# 执行抽样
Z_part = np.zeros(2**(n-m), int)
basis_part = np.arange(0, 2**(n-m))
sample_part = random.choices(basis_part, weights=prob_part, k=N_part)
for i in sample_part:
    Z_part[i] += 1

# 将抽样结果汇总到 0 号进程
Z = np.zeros(2**n, int)
comm.Gatherv(Z_part, Z)

comm.Barrier()

# 由 0 号进程汇总
if rank == 0:
    print(Z)
    print(sum(Z))
    plt.bar(range(2**n), Z)
    plt.show()

MPI.Finalize()

