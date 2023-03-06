import networkx as nx
from Qcover.core import Qcover
from Qcover.backends import CircuitByQulacs
from Qcover.optimizers import COBYLA

ising_g = nx.Graph()
nodes = [(0, 3), (1, 2), (2, 1), (3, 1)]
edges = [(0, 1, 1), (0, 2, 1), (3, 1, 2), (2, 3, 3)]
for nd in nodes:
   u, w = nd[0], nd[1]
   ising_g.add_node(int(u), weight=int(w))
for ed in edges:
    u, v, w = ed[0], ed[1], ed[2]
    ising_g.add_edge(int(u), int(v), weight=int(w))

p = 2
optc = COBYLA(options={'tol': 1e-3, 'disp': True})
ts_bc = CircuitByQulacs()
qc = Qcover(ising_g, p=p, optimizer=optc, backend=ts_bc)
res = qc.run()
print("the result of problem is:\n", res)
qc.backend.optimization_visualization()