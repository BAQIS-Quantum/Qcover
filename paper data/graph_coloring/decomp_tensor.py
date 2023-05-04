from tests.core_without_RQAOA import *
import os

from Qcover.backends import CircuitByTensor
from Qcover.applications.graph_color import GraphColoring

from time import time
import numpy as np
import h5py

import quimb as qu
import quimb.tensor as qtn


def qaoa_tensor(graph, p, params):


    N = len(graph.nodes)
    circ = qu.tensor.Circuit(N)

    for i in graph.nodes():
        circ.apply_gate('H', i)

    for k in range(p):
        for i in graph.nodes:
            node_weight = graph.nodes[i]['weight']
            # print('ndw_%i' % node_weight)

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


# num_nodes_list = np.arange(10,500,40)
num_nodes_list = np.array([4,6])
p = 1
cln = 3
nd = 3
opt = 'greedy'
max_step = 1
time_qcover_tensor = np.zeros(len(num_nodes_list), dtype=float)

cy_ind = 0
for num_nodes in num_nodes_list:

    gct = GraphColoring(node_num=num_nodes, color_num=cln, node_degree=nd)
    g = gct.run()

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
    filename = '../data/graphcolor_decomp_tensor_p%i_nodesnum%i_nd%i_cln%i.h5'%(p, num_nodes_list[0],nd,cln)
else:
    filename = '../data/graphcolor_decomp_tensor_p%i_nd%i_cln%i.h5'%(p,nd,cln)
data = h5py.File(filename, 'w')
data['time_qcover_tensor'] = time_qcover_tensor
data['num_nodes_list'] = num_nodes_list
data['maxiter'] = max_step
data['p'] = p
data['nd'] = nd
data['cln'] = cln
data.close()




