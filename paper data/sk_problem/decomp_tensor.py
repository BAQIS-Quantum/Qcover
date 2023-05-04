import os

from tests.core_without_RQAOA import *

from Qcover.backends import CircuitByTensor
from Qcover.applications.sherrington_kirkpatrick import SherringtonKirkpatrick


from time import time
import numpy as np
import h5py

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
    g = sk.run()

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
    filename = '../data/sk_decomp_tensor_p%i_nodesnum%i.h5'%(p, num_nodes_list[0])
else:
    filename = '../data/sk_decomp_tensor_p%i.h5'%p
data = h5py.File(filename, 'w')
data['time_qcover_tensor'] = time_qcover_tensor
data['num_nodes_list'] = num_nodes_list
data['maxiter'] = max_step
data['p'] = p
data.close()




