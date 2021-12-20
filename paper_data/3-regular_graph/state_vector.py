import sys
sys.path.append(r'../../')
from core import *
from frameworks import CircuitByQiskit
from applications.max_cut import MaxCut

from time import time
import numpy as np
import h5py

from datetime import datetime

import networkx as nx
from scipy.optimize import minimize, rosen, rosen_der


nd = 3
p = 1
# num_nodes_list = np.array([10,14,26, 50,100,300,500, 800, 1000,10000])
num_nodes_list = np.array([10,14])
time_statevector = np.zeros(len(num_nodes_list), dtype=float)
exp_statevector = np.zeros_like(time_statevector)
parametr_f_statevector = np.zeros([len(num_nodes_list),2, p], dtype=float)

cy_ind = 0
max_step = 1

for num_nodes in num_nodes_list:
    mxt = MaxCut(node_num = num_nodes, node_degree=nd)
    mc_mat = nx.adj_matrix(mxt.graph).A
    g = mxt.run()

    gamma = np.random.rand(p)
    beta = np.random.rand(p)

    qser = QCover(g, p,gamma,beta,expectation_calc_method="statevector",
                  computing_framework="qiskit",maxiter=max_step)
    st = time()
    qser_sv = qser.run()
    time_statevector[cy_ind] = time() - st
    exp_statevector[ cy_ind] = qser_sv.fun

    parametr_f_statevector[cy_ind,0,:] = qser_sv.x[0,:]
    parametr_f_statevector[cy_ind,1,:] = qser_sv.x[1,:]

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
data['exp_statevector']  = exp_statevector
data['parametr_f_statevector'] = parametr_f_statevector
data['num_nodes_list'] = num_nodes_list
data['maxiter'] = max_step
data['p'] = p
data['nd'] = nd
data.close()

