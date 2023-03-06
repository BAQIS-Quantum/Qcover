from Qcover.core import Qcover
from Qcover.backends import CircuitByQulacs
from Qcover.optimizers import COBYLA
from Qcover.compiler import CompilerForQAOA
import networkx as nx
import matplotlib.pyplot as plt

# Qcover supports real quantum computers to solve combinatorial optimization problems.
# You only need to transform the combinatorial optimization problem into a weight graph,
# and you can use the quafu quantum computing cloud platform  (http://quafu.baqis.ac.cn/)
# to solve the corresponding problem. The following is an example of a max-cut problem.

# The weight graph corresponding to the combinatorial optimization problem and transformed it to networkx format.
nodes = [(0, 1), (1, 3), (2, 2), (3, 1), (4, 0), (5, 3)]
edges = [(0, 1, -1), (1, 2, -4), (2, 3, 2), (3, 4, -2), (4, 5, -1), (1, 3, 0), (2, 4, 3)]
graph = nx.Graph()
for nd in nodes:
    u, w = nd[0], nd[1]
    graph.add_node(int(u), weight=int(w))
for ed in edges:
    u, v, w = ed[0], ed[1], ed[2]
    graph.add_edge(int(u), int(v), weight=int(w))


# draw weighted graph to be calculated
new_labels = dict(map(lambda x: ((x[0], x[1]), str(x[2]['weight'])), graph.edges(data=True)))
pos = nx.spring_layout(graph)
# pos = nx.circular_layout(g)
nx.draw_networkx(graph, pos=pos, node_size=400, font_size=13, node_color='y')
# nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=new_labels, font_size=15)
nx.draw_networkx_edges(graph, pos, width=2, edge_color='g', arrows=False)
plt.show()

# Using Qcover to calculate the optimal parameters of QAOA circuit.
p = 1
bc = CircuitByQulacs()
optc = COBYLA(options={'tol': 1e-3, 'disp': True})
qc = Qcover(graph, p=p, optimizer=optc, backend=bc)
res = qc.run()
optimal_params = res['Optimal parameter value']

# Compile and send the QAOA circuit to the quafu cloud.
# Token parameter should be set according to your own account
# For more introduction see https://github.com/ScQ-Cloud/pyquafu
token = "IB2Vz-3bqNNRHPCIw1RcdBMgPq8LNeGZe4KbBYDE_0A.9BzN5MjMwkjN2EjOiAHelJCLzITM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye"
cloud_backend = 'ScQ-P20'
qcover_compiler = CompilerForQAOA(graph, p=p, optimal_params=optimal_params, apitoken=token, cloud_backend=cloud_backend)
task_id = qcover_compiler.send(wait=True, shots=5000, task_name='MaxCut')
# If you choose wait=True, you have to wait for the result to return.
# If you choose wait=False, you can execute the following command to query the result status at any time,
# and the result will be returned when the task is completed.
quafu_solver = qcover_compiler.task_status_query(task_id)
if quafu_solver:
    counts_energy = qcover_compiler.results_processing(quafu_solver)
    qcover_compiler.visualization(counts_energy)

