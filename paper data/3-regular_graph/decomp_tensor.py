from tests.core_without_RQAOA import *
import os

from Qcover.backends import CircuitByTensor
from Qcover.applications.max_cut import MaxCut


from time import time
import numpy as np
import h5py

import networkx as nx

nd = 3
p = 1
#num_nodes_list = np.array([10, 14, 26, 50, 100,300,500,800,1000])
num_nodes_list = np.array([10,12])
time_qcover_tensor = np.zeros(len(num_nodes_list), dtype=float)
exp_qcover_tensor = np.zeros_like(time_qcover_tensor)
parametr_f_qcovertensor = np.zeros([len(num_nodes_list),2, p], dtype=float)

cy_ind = 0
max_step = 1  
for num_nodes in num_nodes_list:
    mxt = MaxCut(node_num = num_nodes, node_degree=nd)
    mc_mat = nx.adj_matrix(mxt.graph).A
    g = mxt.run()

    #for the p = 4, we choose contract_opt = 'greedy-rf'
    quimb_bc = CircuitByTensor(contract_opt='greedy')
    optc = COBYLA(maxiter=1, tol=1e-6, disp=True)
    qc = Qcover(g, p=p, optimizer=optc, backend=quimb_bc)
    st = time()
    res = qc.run()
    time_qcover_tensor[cy_ind] = time() - st

    cy_ind += 1
dirs = '../data'

if not os.path.exists(dirs):
    os.makedirs(dirs)

if len(num_nodes_list) == 1:
    filename = '../data/Qcover_decomp_tensor_p%i_nd%i_nodesnum%i.h5'%(p, nd, num_nodes_list[0])
else:
    filename = '../data/Qcover_decomp_tensor_p%i_nd%i.h5'%(p, nd)
data = h5py.File(filename, 'w')
data['time_qcover_tensor'] = time_qcover_tensor
data['maxiter'] = max_step
data['nd'] = nd
data['p'] = p
data.close()



