import time
from Qcover import COBYLA
from Qcover.applications import MaxCut
from Qcover.backends import CircuitByQulacs
from Qcover.core import Qcover

T = 20
p = 1
t_qaoa = []
t_rqaoa = []
exp_qaoa = []
exp_rqaoa = []

for iter in range(T):
    mxt = MaxCut(node_num=100, node_degree=3)
    ising_g, shift = mxt.run()
    qc = Qcover(ising_g, p,
                optimizer=COBYLA(options={'tol': 1e-6, 'disp': False}),
                backend=CircuitByQulacs())

    st = time.time()
    sol = qc.run(mode='QAQA')
    ed = time.time()
    t_qaoa.append(ed - st)
    exp_qaoa.append(sol["Expectation of Hamiltonian"])
    print("time cost by QAOA is:", ed - st)
    print("expectation value by QAOA is:", sol["Expectation of Hamiltonian"])

    res_g = ising_g.copy()
    rqc = Qcover(ising_g, p,
                optimizer=COBYLA(options={'tol': 1e-6, 'disp': False}),
                backend=CircuitByQulacs())

    st = time.time()
    sol = rqc.run(mode='RQAQA', node_threshold=1)
    ed = time.time()
    t_rqaoa.append(ed - st)

    exph = 0
    for (x, y) in res_g.nodes.data('weight', default=0):
        exph = exph + y * (sol[x] * 2 - 1)
    for (u, v, c) in res_g.edges.data('weight', default=0):
        exph = exph + c * (sol[u] * 2 - 1) * (sol[v] * 2 - 1)

    exp_rqaoa.append(exph)
    print("time cost by RQAOA is:", ed - st)
    print("expectation value by RQAOA is:", exph)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(T), t_qaoa, "ob-", label="QAOA")
plt.plot(range(T), t_rqaoa, "^r-", label="RQAOA")
plt.ylabel('Time cost')
plt.xlabel('iteration id')
plt.title("comparison of time taken by QAOA with RQAOA")
plt.legend()
plt.savefig('E:/Working_projects/QAOA/QCover/result_log/maxcut_time_large.png')  # maxcut_serial
plt.show()

plt.figure(1)
plt.plot(range(T), exp_qaoa, "ob-", label="QAOA")
plt.plot(range(T), exp_rqaoa, "^r-", label="RQAOA")
plt.ylabel('Expectation value')
plt.xlabel('iteration id')
plt.title("comparison of expectation value calculated by QAOA with RQAOA")
plt.legend()
plt.savefig('/public/home/humengjun/Qcover/result_log/tc.png')
plt.show()