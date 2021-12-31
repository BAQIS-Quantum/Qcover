# from Qcover import *
import sys
sys.path.append(r'E:\Working_projects\QAOA\Qcove\Qcover|')
import time
import networkx as nx
from ..applications import MaxCut

G = nx.Graph()
n = 0
i = 0
str = input("输入文件名：")
with open(str) as f:
    for line in f:
        if line.startswith('#'):
            continue
        data = line.split()
        if i == 0:
            n = data[0]
            G.add_nodes_from(range(1, int(n) + 1), weight=1)
        else:
            p = data[0]
            q = data[1]
            r = data[2]
            G.add_edge(int(p), int(q), weight=int(r))
        i = i + 1

mxt = MaxCut(G)
ising_g = mxt.run()
p = 1
from ..optimizers import GradientDescent, Interp, Fourier, COBYLA
optc = COBYLA(maxiter=30, tol=1e-6, disp=True)
from ..backends import CircuitByQiskit, CircuitByCirq, CircuitByQulacs, CircuitByProjectq, CircuitByTensor

qiskit_bc = CircuitByQulacs()
from ..core import Qcover
# qser_sta = Qcover(ising_g, p=1, expectation_calc_method="statevector")   #qulacs, backend_name="qulacs"#  #cirq tket  projectq
qser_sta = Qcover(ising_g, p,
                  optimizer=optc,
                  backend=qiskit_bc)
st = time.time()
res_sta_ip = qser_sta.run(is_parallel=False)  # True
ed = time.time()
print("time cost:", ed - st)
