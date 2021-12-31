import os
from Qcover.core import *
import cotengra as ctg

from Qcover.backends import CircuitByTensor
from Qcover.applications.sherrington_kirkpatrick import SherringtonKirkpatrick


from time import time
import numpy as np
from datetime import datetime
import h5py

import quimb as qu
import quimb.tensor as qtn
import networkx as nx
from scipy.optimize import minimize, rosen, rosen_der


def qaoa_tensor(graph, p, params):
    N = len(graph.nodes)
    circ = qu.tensor.Circuit(N)

    for i in graph.nodes():
        circ.apply_gate('H', i)

    for k in range(p):
        for i in graph.nodes:
            node_weight = graph.nodes[i]['weight']

            circ.apply_gate('rz', 2 * params[2 * k] * node_weight, i)

        for edge in graph.edges:
            edge_weight = graph.get_edge_data(edge[0], edge[1])['weight']

            gamma = -params[2 * k] * edge_weight
            circ.apply_gate('RZZ', gamma, edge[0], edge[1])

        for i in graph.nodes:
            circ.apply_gate('rx', 2 * params[2 * k + 1], i)
    return circ


def expectation(mx_g, circ, opt):
    expectation = 0
    ZZ = qu.pauli('Z') & qu.pauli('Z')
    for node in mx_g.nodes:
        w = mx_g.nodes[node]['weight']
        expectation = w * circ.local_expectation(qu.pauli('Z'), node, optimize=opt) + expectation

    for edge in mx_g.edges:
        w = mx_g.get_edge_data(edge[0], edge[1])['weight']
        expectation = w * circ.local_expectation(ZZ, edge, optimize=opt) + expectation
    return expectation.real


def energy(params, mx_g, p,opt):
    circ = qaoa_tensor(mx_g, p, params)
    expec = expectation(mx_g, circ,opt)
    return expec



p = 1
opt = 'greedy'
#num_nodes_list = np.arange(4,50,4)
num_nodes_list = np.array([10,12])
time_tensor = np.zeros(len(num_nodes_list), dtype=float)


cy_ind = 0
max_step = 1
for num_nodes in num_nodes_list:
    sk = SherringtonKirkpatrick(num_nodes)
    sk_graph = sk.run()

    gamma = np.random.rand(p)
    beta = np.random.rand(p)

    st = time()
    qser_whole_tensor = minimize(energy, np.asarray([gamma, beta]), args=(sk_graph, p,opt),
                           method='COBYLA',
                           tol=1e-14,
                           jac=rosen_der,
                           options={'gtol': 1e-8, 'maxiter': max_step, 'disp': True})
    time_tensor[cy_ind] = time() - st
    cy_ind += 1

dirs = '../data'
if not os.path.exists(dirs):
    os.makedirs(dirs)

if len(num_nodes_list) == 1:
    filename = '../data/sk_wtensor_p%i_nodesnum%i.h5'%(p, num_nodes_list[0])
else:
    filename = '../data/sk_wtensor_p%i.h5'%(p)
data = h5py.File(filename, 'w')
data['time_tensor'] = time_tensor
data['num_nodes_list'] = num_nodes_list
data['maxiter'] = max_step
data['p'] = p
data.close()






