
from Qcover.core import Qcover
from Qcover.backends import CircuitByCirq, CircuitByProjectq, CircuitByQulacs
from Qcover.optimizers import COBYLA
import networkx as nx
import matplotlib.pyplot as plt

p = 1
g = nx.Graph()

node_num, edge_num = 6, 7
# nodes, edges = Qcover.generate_graph_data(node_num, edge_num)

nodes = {(0, 5), (1, 1), (2, 8), (3, 3)}
edges = {(0, 1, 1), (1, 2, 1), (3, 0, 6)}

# nodes = {(0,1), (1,3), (2,2), (3,1), (4,0), (5,3)}
# edges = {(0,1,-1), (1,2,-4), (2,3,2), (3,4,-2), (4,5,-1), (1,3,0), (2,4,3)}

for nd in nodes:
    u, w = nd[0], nd[1]
    g.add_node(int(u), weight=int(w))
for ed in edges:
    u, v, w = ed[0], ed[1], ed[2]
    g.add_edge(int(u), int(v), weight=int(w))


# bc = CircuitByQiskit()
# bc = CircuitByProjectq()
# bc = CircuitByCirq()
# bc = CircuitByQton()
bc = CircuitByQulacs()
# bc = CircuitByTensor()
optc = COBYLA(options={'tol': 1e-3, 'disp': True})
qc = Qcover(g, p=p, optimizer=optc, backend=bc)
res = qc.run()
optimal_params = res['Optimal parameter value']
print('optimal_params:', optimal_params)

# draw weighted graph
new_labels = dict(map(lambda x: ((x[0], x[1]), str(x[2]['weight'])), g.edges(data=True)))
pos = nx.spring_layout(g)
# pos = nx.circular_layout(g)
nx.draw_networkx(g, pos=pos, node_size=400, font_size=13, node_color='y')
# nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=new_labels, font_size=15)
nx.draw_networkx_edges(g, pos, width=2, edge_color='g', arrows=False)
plt.show()



# compiler and send to quafu cloud
from Qcover.compiler import CompilerForQAOA
BD_token = "QU9FimE2tOuY_AtN0QvwwjbNqeeMR-4jWWwXXQ9mRFh.9lTNwgzN4cjN2EjOiAHelJCLzITM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye"
# token = "EE4QgBwaMIKF5w2xDg_pclZeDGKTAJZZq4sN6aKKdqC.QfxYzM0MDO1YjNxojIwhXZiwiI4YjI6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye"
cloud_backend = "ScQ-P10"
shots = 5000
qaoa_compiler = CompilerForQAOA(g, p=p, optimal_params=optimal_params, apitoken=BD_token, cloud_backend=cloud_backend)
quafu_solver = qaoa_compiler.run(shots=shots)
counts_energy = qaoa_compiler.results_processing(quafu_solver)
qaoa_compiler.visualization(counts_energy)
