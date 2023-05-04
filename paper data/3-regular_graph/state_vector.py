import os
from tests.core_without_RQAOA import *
from Qcover.backends import CircuitByQiskit
from Qcover.applications.max_cut import MaxCut

from time import time
import numpy as np
import h5py

import networkx as nx

nd = 3
p = 1
# num_nodes_list = np.array([10,14,26, 50,100,300,500, 800, 1000,10000])
num_nodes_list = np.array([10,14])
time_statevector = np.zeros(len(num_nodes_list), dtype=float)
cy_ind = 0
max_step = 1

for num_nodes in num_nodes_list:
    mxt = MaxCut(node_num = num_nodes, node_degree=nd)
    mc_mat = nx.adj_matrix(mxt.graph).A
    g = mxt.run()

    quimb_bc = CircuitByQiskit(expectation_calc_method="statevector")
    optc = COBYLA(maxiter=1, tol=1e-6, disp=True)
    qc = Qcover(g, p=p, optimizer=optc, backend=quimb_bc)
    st = time()
    res = qc.run()
    time_statevector[cy_ind] = time() - st
    cy_ind += 1

dirs = '../data'
if not os.path.exists(dirs):
    os.makedirs(dirs)

if len(num_nodes_list) == 1:
    filename = '../data/statevector_p%i_nd%i_nodesnum%i.h5'%(p, nd, num_nodes_list[0])
else:
    filename = '../data/stetevector_p%i_nd%i.h5'%(p, nd)
data = h5py.File(filename, 'w')

data['time_statevector'] = time_statevector
data['num_nodes_list'] = num_nodes_list
data['maxiter'] = max_step
data['p'] = p
data['nd'] = nd
data.close()

