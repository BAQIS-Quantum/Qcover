
# if __name__ == "__main__":
from Qcover.core import Qcover
from Qcover.backends import CircuitByCirq
from Qcover.optimizers import COBYLA
import networkx as nx
import matplotlib.pyplot as plt

# generate random weight graph
node_num, edge_num = 5, 6
nodes, edges = Qcover.generate_graph_data(node_num, edge_num)

# Run qcover to generate optimal parameters
p = 1
g = Qcover.generate_weighted_graph(nodes, edges)
# qulacs_bc = CircuitByQiskit()
# qulacs_bc = CircuitByProjectq()
qulacs_bc = CircuitByCirq()
# qulacs_bc = CircuitByQton()
# qulacs_bc = CircuitByQulacs()
# qulacs_bc = CircuitByTensor()
optc = COBYLA(options={'tol': 1e-3, 'disp': True})
qc = Qcover(g, p=p, optimizer=optc, backend=qulacs_bc)
res = qc.run()
optimal_params = res['Optimal parameter value']
print('optimal_params:', optimal_params)

# draw weighted graph
new_labels = dict(map(lambda x: ((x[0], x[1]), str(x[2]['weight'])), g.edges(data=True)))
pos = nx.spring_layout(g)
nx.draw_networkx(g, pos=pos, node_size=500, font_size=15, node_color='y')
# nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=new_labels, font_size=15)
nx.draw_networkx_edges(g, pos, width=2, edge_color='g', arrows=False)
plt.show()


# compiler and send to quafu cloud
from Qcovercompiler import CompilerforQAOA
token = "cJNxO0iQ708Nt61AO_NqRBTv-v5NyBMW_GmmV5decbC.Qf0YTOwETM1YjNxojIwhXZiwiI4YjI6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye"
cloud_backend = "ScQ-P20"
shots = 1024
qaoa_compiler = CompilerforQAOA(g, p=p, optimal_params=optimal_params, apitoken=token, cloud_backend=cloud_backend)
solver = qaoa_compiler.run(shots=shots)
counts_energy = qaoa_compiler.results(solver)
qaoa_compiler.visualization(counts_energy)

