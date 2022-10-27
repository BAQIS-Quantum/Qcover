import json
import networkx as nx
import matplotlib.pyplot as plt
from Qcover.compiler.hardware_library import BuildLibrary
from quafu import User
import ast


# build library
user = User()
user.save_apitoken("IB2Vz-3bqNNRHPCIw1RcdBMgPq8LNeGZe4KbBYDE_0A.9BzN5MjMwkjN2EjOiAHelJCLzITM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
# user.save_apitoken("IK5TmYrX6v3gIt_AMW4JsQKHWH_-ZrFQSY793FZiZoH.QfyUDN3IzN1YjNxojIwhXZiwiI4YjI6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
backend = "ScQ-P10"
BL = BuildLibrary(backend=backend, fidelity_threshold=96)
BL.build_substructure_library()


# test read library
with open('LibSubstructure_'+backend+'.txt', 'r', encoding='utf-8') as f:
    substructure_data = eval(f.read())


sub_G = nx.MultiDiGraph()
sub_G.add_edges_from(substructure_data['substructure_dict'][4][0])
pos = nx.spring_layout(sub_G)
nx.draw(sub_G, pos=pos, with_labels=True, node_color='r', node_size=500, edge_color='b', width=1, font_size=16)
plt.show()



# # new code add linear coupling
# qubits = 5
# sub_clist = [qubits[0:2] for qubits in substructure_data['substructure_dict'][qubits][0]]
# for sub in substructure_data['substructure_dict'][qubits]:
#     sublist = [qubits[0:2] for qubits in sub]
#     G = nx.Graph()
#     G.add_edges_from(sublist)
#     node_degree = dict(G.degree)
#     sort_degree = sorted(node_degree.items(), key=lambda kv: kv[1], reverse=True)
#     if sort_degree[0][1]==2:
#         print('subclist',sub)
#         sub_clist = sublist
#         break
# G = nx.Graph()
# G.add_edges_from(sub_clist)
# pos = nx.spring_layout(G)
# nx.draw(G, pos=pos, with_labels=True, node_color='r', node_size=150, edge_color='b',
#         width=1, font_size=9)
# plt.show()
# # # new code add linear coupling