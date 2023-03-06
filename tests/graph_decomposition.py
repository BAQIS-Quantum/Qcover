# -*- coding: utf-8 -*-
"""
Created on Wed Jul  12 10:12:33 2021

@author: YaNan Pu
"""
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool
import warnings
from collections import defaultdict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn, CircuitOp, CircuitStateFn, \
    MatrixExpectation, X, Y, Z, I


def generate_graph_data(node_num, edge_num):
    '''
    generate a simple graph‘s weights of nodes and edges with node number is node_num, edge number is edge_num
    Args:
        node_num (int): node number in graph
        edge_num (int): edge number in graph
    Return:
        nodes(set of tuple(nid, node_weight) ), edges(set of tuple(nid1, nid2, edge_weight))'''

    nodes = set()
    for i in range(node_num):
        ndw = np.random.choice(range(10))
        nodes |= {(i, ndw)}

    edges = set()
    cnt = edge_num
    max_edges = node_num * (node_num - 1) / 2
    if cnt > max_edges:
        cnt = max_edges
    while cnt > 0:
        u = np.random.randint(node_num)
        v = np.random.randint(node_num)
        if u == v:  # without self loop
            continue
        flg = 0
        for e in edges:     # without duplicated edges
            if set(e[:2]) == set([v, u]):
                flg = 1
                break
        if flg == 1:
            continue
        edw = np.random.choice(range(10))  # assign random weights to edges
        edges |= {(u, v, edw)}
        # edges.append([u, v, c])
        cnt -= 1
    return nodes, edges


def generate_weighted_graph(node_set, edge_set):
    '''generate a graph with node-weight map and edge-weight map
    Args:
        node_set (set): node-weight map, element form is tuple(nid, weight)
        edge_set (set): edge-weight map, element form is tuple(nid1, nid2, edge_weight)
    Return:
        graph (networkx): graph generated from args
        '''
    g = nx.Graph()
    for item in node_set:
        g.add_node(item[0], weight=item[1])

    for item in edge_set:
        g.add_edge(item[0], item[1], weight=item[2])

    return g


def get_graph_weights(graph):
    '''
    get the weights of nodes and edges
    :param graph: graph (networkx)
    :return: node weights form is, edges weights form is
    '''
    nodew = nx.get_node_attributes(graph, 'weight')
    edw = nx.get_edge_attributes(graph, 'weight')
    edgew = edw.copy()
    for key, val in edw.items():
        edgew[(key[1], key[0])] = val

    return nodew, edgew


def generate_subgraph(graph, dtype, p=1):
    '''
    according to the arguments of dtype and p to generate subgraphs from graph
    Args:
        graph (nx.Graph()): graph to be decomposed
        dtype (string): set node or edge, the ways according to which to decompose the graph
        p (int): the hop of subgraphs
    Return:
        subg_dict (dict) form as {node_id : subg, ..., (node_id1, node_id2) : subg, ...}
    '''

    if dtype not in ['node', 'edge']:
        print("Error: wrong dtype, dtype should be node or edge")
        return None

    subg_dict = defaultdict(list)
    nodew, edgew = get_graph_weights(graph)

    if dtype == 'node':
        for node in graph.nodes:
            node_set = {(node, nodew[node])}   #
            edge_set = set()
            for i in range(p):
                # for nd1 in nodes:
                #     for nd2 in graph[nd1[0]]:
                #         print(type(nd2))
                new_nodes = {(nd2, nodew[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}  #graph.adj[nd1]
                # for nd1 in node_set:
                #     for nd2 in graph[nd1[0]]:
                #         print("the weights of edge (%s, %s) is %s" % (str(nd1[0]), str(nd2), str(edgew[nd1[0], nd2])))
                new_edges = {(nd1[0], nd2, edgew[nd1[0], nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                node_set |= new_nodes
                edge_set |= new_edges

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[node] = subg
    else:
        for edge in graph.edges:
            node_set = {(edge[0], nodew[edge[0]]), (edge[1], nodew[edge[1]])}
            edge_set = {(edge[0], edge[1], edgew[edge[0], edge[1]])}  # set()

            for i in range(p):
                new_nodes = {(nd2, nodew[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                new_edges = {(nd1[0], nd2, edgew[nd1[0], nd2]) for nd1 in node_set for nd2 in graph.adj[nd1[0]]}  # graph.adj[nd1][nd2]['weight']
                node_set |= new_nodes
                edge_set |= new_edges  # find any adj edges, take union of set to get rid of duplicate node

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[edge] = subg

    return subg_dict


def graph_decomposision(graph, p=1):
    '''
    according to dtype to decompose graph
    Args:
        graph (nx.Graph()): graph to be composed
        p (int): the hop of subgraphs'''

    if p <= 0:
        warnings.warn(" the argument of p should be >=1 in qaoa problem, so p would be seted to the default value at 1")
        p = 1
        # return None

    subg_node = generate_subgraph(graph, "node", p)    #, subg_setv
    subg_edge = generate_subgraph(graph, "edge", p)    #, subg_sete
    return [subg_node, subg_edge]           #, [subg_setv, subg_sete]


def graph_to_circuit(graph, gamma, beta, original_e=None):
    '''
    transform the graph to circuit
    Args:
        graph (nx.Graph()): graph to be transformed to circuit
        gamma (float): Optimal parameters
        beta (float): Optimal parameters
        original_e (Optional[None, int, tuple])
    Return:
        if original_e=None, then the graph is the whole original graph generated by
        generate_weighted_graph(), so just return the circuit transformed by it

        if original_e is a int, then the subgraph is generated by node(idx = original_e
        in whole graph), so return the it's idx mapped by node_to_qubit[], and the circuit

        if original_e is a tuple, then the subgraph is generated by edge(node idx = original_e
        in whole graph), so return the it's idx mapped by node_to_qubit[] as
        tuple(mapped node_id1, mapped node_id2), and the circuit
        '''

    circ_qreg = QuantumRegister(len(graph.nodes), 'q')
    circ = QuantumCircuit(circ_qreg)

    nodew, edgew = get_graph_weights(graph)

    # generate node_to_qubit[] map from node id in graph to [0, node_num in graph - 1]
    # so the subgraph would has it's own qubits number
    node_to_qubit = defaultdict(int)
    node_list = list(graph.nodes)
    for i in range(len(node_list)):
        node_to_qubit[node_list[i]] = i

    # generate circuit from graph
    for i in graph.nodes:
        u = node_to_qubit[i]
        circ.h(u)
        circ.rz(2*gamma * nodew[i], u)
    circ.barrier()

    for edge in graph.edges:
        u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
        circ.rzz(2*gamma * edgew[edge[0], edge[1]], u, v)
        # circ.cp(-2 * gamma, u, v)
        # circ.p(gamma, u)
        # circ.p(gamma, v)

    circ.barrier()
    for i in graph.nodes:
        circ.rx(2 * beta, node_to_qubit[i])
        # circ.measure(i, i)

    if original_e is None:
        return original_e, circ
    elif isinstance(original_e, int):
        return node_to_qubit[original_e], circ
    else:
        return (node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]), circ


def get_statevector_expectation(subc, *nodes):
    qubit_num = subc.num_qubits
    subc.save_statevector()  # Tell simulator to save statevector
    # subc.draw(output='mpl', interactive=True)

    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(subc)  # Create a Qobj from the circuit for the simulator to run
    result = sim.run(qobj).result()
    out_state = result.get_statevector()

    op = 1
    for i in range(qubit_num):
        if i in nodes:
            op = np.kron(op, Z.to_matrix())
        else:
            op = np.kron(op, I.to_matrix())

    exp_res = np.matmul(np.matmul(out_state.conj().T, op), out_state)

    print('Expectation_state Value of subgraph generate by %s = %lf' % (str(nodes), exp_res.real))
    return exp_res.real


def get_sample_expectation(subc, *nodes):
    qubit_num = subc.num_qubits

    op = Z if 0 in nodes else I
    for i in range(1, qubit_num):
        if i in nodes:
            op = op ^ Z
        else:
            op = op ^ I

    # for small circuit
    # psi = CircuitStateFn(subc)
    # exp_res = psi.adjoint().compose(op).compose(psi).eval().real

    subc = CircuitStateFn(subc)
    backend = Aer.get_backend('qasm_simulator')
    q_instance = QuantumInstance(backend, shots=100024)
    measurable_expression = StateFn(op, is_measurement=True).compose(subc)
    expectation = PauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(q_instance).convert(expectation)
    exp_res = sampler.eval().real
    print('Expectation_sample Value of subgraph generate by %s = %lf' % (str(nodes), exp_res))
    return exp_res


def expectation_calculation(sub_circ_list):
    res_state, res_sample = 0, 0
    for key, val in sub_circ_list.items():
        # print("draw the subcircuit generated by nodes ", key)
        # val[1].draw(output='mpl', interactive=True)
        # wires = list(val[0])
        wires = [val[0]] if isinstance(val[0], int) else list(val[0])
        res_sample += val[1] * get_sample_expectation(val[2], *wires)
        res_state += val[1] * get_statevector_expectation(val[2], *wires)

    return [res_sample, res_state]


def expectation_calculation_parallel(subc, func_type):

    res_expectation = 0
    wires = [subc[0]] if isinstance(subc[0], int) else list(subc[0])
    if func_type == 'sample':
        res_expectation = subc[1] * get_sample_expectation(subc[2], *wires)
    else:
        res_expectation = subc[1] * get_statevector_expectation(subc[2], *wires)

    return res_expectation


if __name__ == "__main__":

    node_num, edge_num = 4, 5
    init_gamma, init_beta = 1.9, 0.5

    nodes, edges = generate_graph_data(node_num, edge_num)
    g = generate_weighted_graph(nodes, edges)
    nx.draw_networkx(g)
    plt.show()

    _, circ = graph_to_circuit(g, init_gamma, init_beta)
    circ.draw(output='mpl', interactive=True) # the qubits is mapped, so isn't corresponding to nodes id the g
    #
    subg_list = graph_decomposision(g, 1)    #, subg_set
    nodew, edgew = get_graph_weights(g)

    subcs = defaultdict(list)
    for i, subg in enumerate(subg_list):
        for key, val in subg.items():
            mk, sub_circ = graph_to_circuit(val, init_gamma, init_beta, key)
            if i == 0:
                # print("draw the subgraphs generated by nodes")
                subcs[key] = [mk, nodew[key], sub_circ]
            else:
                # print("draw the subgraphs generated by edges")
                subcs[key] = [mk, edgew[key[0], key[1]], sub_circ]

            # tp = "node" if i == 0 else "edge"
            # lgd = "sub graph generate by " + tp + " " + str(key)
            # nx.draw_networkx(val)
            # plt.title(lgd)
            # plt.show()
            #
            # sub_circ.draw(output='mpl', interactive=True)

        # sub_circ_list.append(subcs)

    # time_start = time.time()
    # Hf1, Hf2 = expectation_calculation(subcs)
    # time_end = time.time()
    # print('without parallel takes: ', time_end - time_start)  # 2.301382064819336
    # print("Total expectation_sample of original graph is: ", Hf1)
    # print("Total expectation_state of original graph is: ", Hf2)

    time_start = time.time()
    # parallel calculate the expectation of every circuit
    pool = Pool()  # without arguments, use all cpu cores
    res1 = pool.starmap(expectation_calculation_parallel, [(subc, 'sample') for subc in subcs.values()])
    res2 = pool.starmap(expectation_calculation_parallel, [(subc, 'statevector') for subc in subcs.values()])
    # res2 = pool.map(expectation_calculation_parallel, subcs.values())
    pool.close()
    pool.join()
    #
    exp1, exp2 = 0, 0
    for i in range(len(res1)):
        exp1 += res1[i]
        exp2 += res2[i]
    print("Total expectation1 of original graph is: ", exp1)
    print("Total expectation2 of original graph is: ", exp2)

    time_end = time.time()
    print('statevector with parallel takes: ', time_end - time_start)  # 45.035425662994385

