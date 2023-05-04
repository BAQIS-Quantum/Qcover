import os
import time
import networkx as nx
import numpy as np
import h5py

from qiskit.optimization.applications.ising import max_cut
from qiskit.aqua import aqua_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit.aqua.algorithms import QAOA
from qiskit import Aer

from Qcover.applications import MaxCut




nd = 3

p = 1

# num_nodes_list = np.array([4,6,8,10,12, 14])
num_nodes_list = np.array([4,6])
time_qiskit = np.zeros_like(num_nodes_list, dtype='float')

# nx.draw_networkx(mxt.graph)
# plt.show()

cy_ind = 0
for num_nodes in num_nodes_list:
    mxt = MaxCut(node_num=num_nodes, node_degree=nd)
    mc_mat = nx.adjacency_matrix(mxt.graph).A
    qubit_op, offset = max_cut.get_operator(mc_mat)  #
    aqua_globals.random_seed = 10598
    qaoa = QAOA(qubit_op, optimizer=COBYLA(maxiter=30), p=1,
            quantum_instance=Aer.get_backend('statevector_simulator'))
    st = time.time()
    result = qaoa.compute_minimum_eigenvalue()
    ed = time.time()
    time_qiskit[cy_ind] = ed - st
    cy_ind += 1

dirs = '../data'
if not os.path.exists(dirs):
    os.makedirs(dirs)
if len(num_nodes_list) == 1:
    filename = '../data/qiskit_p%i_nd%i_nodesnum%i.h5' % (p, nd, num_nodes_list[0])
else:
    filename = '../data/qiskit_p%i_nd%i.h5' % (p, nd)
    data = h5py.File(filename, 'w')

data['time_qiskit'] = time_qiskit
data['p'] = p
data['num_nodes_list'] = num_nodes_list
data['nd'] = nd
