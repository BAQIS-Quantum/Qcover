import os
import sys
sys.path.append(r'../../')

from core import *
import cotengra as ctg


from frameworks import CircuitByQiskit
from frameworks import CircuitByTensor
from applications.sherrington_kirkpatrick import SherringtonKirkpatrick


from time import time
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
# import ast


import quimb as qu
import quimb.tensor as qtn
import networkx as nx
from scipy.optimize import minimize, rosen, rosen_der


p = 1
opt = 'greedy'
# num_nodes_list = np.arange(4,64,4)
num_nodes_list = np.array([10,12])
time_qcover_tensor = np.zeros(len(num_nodes_list), dtype=float)
exp_qcover_tensor = np.zeros_like(time_qcover_tensor)
parametr_f_qcovertensor = np.zeros([len(num_nodes_list),2, p], dtype=float)

cy_ind = 0
max_step = 1
for num_nodes in num_nodes_list:

    sk = SherringtonKirkpatrick(num_nodes)
    sk_graph = sk.run()

    gamma = np.random.rand(p)
    beta = np.random.rand(p)

    qser = QCover(sk_graph, p, gamma, beta, computing_framework="quimb", contract_opt = opt,maxiter = max_step)
    st = time()
    qser_tensor = qser.run()
    time_qcover_tensor[cy_ind] = time() - st

    exp_qcover_tensor[cy_ind] = qser_tensor.fun
    parametr_f_qcovertensor[cy_ind,0,:] = qser_tensor.x[0,:]
    parametr_f_qcovertensor[cy_ind,1,:] = qser_tensor.x[1,:]

    cy_ind += 1
dirs = '../data'

if not os.path.exists(dirs):
    os.makedirs(dirs)

if len(num_nodes_list) == 1:
    filename = '../data/sk_decomp_tensor_p%i_nodesnum%i.h5'%(p, num_nodes_list[0])
else:
    filename = '../data/sk_decomp_tensor_p%i.h5'%(p)
data = h5py.File(filename, 'w')
data['time_qcover_tensor'] = time_qcover_tensor
data['exp_qcover_tensor'] = exp_qcover_tensor
data['parametr_f_qcovertensor'] = parametr_f_qcovertensor
data['num_nodes_list'] = num_nodes_list
data['maxiter'] = max_step
data['p'] = p
data.close()




