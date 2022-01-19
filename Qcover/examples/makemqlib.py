import time
import networkx as nx
from Qcover.applications import MaxCut
from Qcover.optimizers import GradientDescent, Interp, Fourier, COBYLA
from Qcover.backends import CircuitByQiskit, CircuitByCirq, CircuitByQulacs, CircuitByProjectq, CircuitByTensor
from Qcover.core import Qcover

G = nx.Graph()
node_num, edge_num = 0, 0
i = 0
# str = input("输入文件名：")
str = "g002201.txt"
with open(str) as f:
    for line in f:
        if line.startswith('#'):
            continue
        data = line.split()
        if i == 0:
            node_num, edge_num = data
            G.add_nodes_from(range(1, int(node_num) + 1))  #, weight=1
        else:
            u, v, w = data
            G.add_edge(int(u), int(v), weight=int(w))
        i = i + 1

p = 5
mxt = MaxCut(G)
ising_g = mxt.run()
if float(edge_num)/float(node_num) > 5.3:
    bc = CircuitByTensor()
else:
    bc = CircuitByQulacs()

optc = COBYLA(options={'tol': 1e-3, 'disp': True})
optf = Fourier(p=p, q=4, r=0, alpha=0.6, optimize_method="COBYLA", options={'tol': 1e-3, 'disp': False})

qc_c = Qcover(ising_g, p,
              optimizer=optc,
              backend=bc)   # ts_bc
st = time.time()
res_c = qc_c.run(is_parallel=False)  # True
ed = time.time()
print("time cost:", ed - st)
print("expectation is:", res_c['Expectation of Hamiltonian'])


qc_f = Qcover(ising_g, p,
              optimizer=optf,
              backend=bc)   # ts_bc

st = time.time()
res_f = qc_f.run(is_parallel=False)  # True
ed = time.time()
print("time cost:", ed - st)
print("expectation is:", res_f['Expectation of Hamiltonian'])
