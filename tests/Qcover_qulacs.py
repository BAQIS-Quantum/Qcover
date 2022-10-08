# -*- coding: utf-8 -*-
"""
Created on Wed Jul  12 10:12:33 2021

@author: YaNan Pu
"""
from cProfile import label
from tokenize import Number
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
import os
from multiprocessing import Pool
import warnings
from collections import defaultdict
from qulacs import Observable,  QuantumCircuit, QuantumState, QuantumStateGpu

global p, element_to_graph, exp_path, nodes_weight, edges_weight, use_gpu

def generate_graph_data(node_num, edge_num):

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

    g = nx.Graph()
    for item in node_set:
        g.add_node(item[0], weight=item[1])

    for item in edge_set:
        g.add_edge(item[0], item[1], weight=item[2])

    return g


def get_graph_weights(graph):

    nodes_weight= nx.get_node_attributes(graph, 'weight')
    edgew = nx.get_edge_attributes(graph, 'weight')
    
    edges_weight = edgew.copy()
    for key, val in edgew.items():
        edges_weight[(key[1], key[0])] = val

    return nodes_weight, edges_weight


def generate_subgraph(graph, dtype):
    global p, nodes_weight, edges_weight
    # print("the value of p is: ", p)

    if dtype not in ['node', 'edge']:
        print("Error: wrong dtype, dtype should be node or edge")
        return None

    subg_dict = defaultdict(list)

    if dtype == 'node':
        for node in graph.nodes:
            node_set = {(node, nodes_weight[node])}   #
            edge_set = set()
            for i in range(p):
                new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}  #graph.adj[nd1]
                new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                node_set |= new_nodes
                edge_set |= new_edges

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[node] = subg
    else:
        for edge in graph.edges:
            node_set = {(edge[0], nodes_weight[edge[0]]), (edge[1], nodes_weight[edge[1]])}
            edge_set = {(edge[0], edge[1], edges_weight[edge[0], edge[1]])}  # set()

            for i in range(p):
                new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in graph.adj[nd1[0]]}  # graph.adj[nd1][nd2]['weight']
                node_set |= new_nodes
                edge_set |= new_edges  # find any adj edges, take union of set to get rid of duplicate node

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[edge] = subg

    return subg_dict


def graph_decompose(graph):
    subg_node = generate_subgraph(graph, "node")    #, subg_setv
    subg_edge = generate_subgraph(graph, "edge")    #, subg_sete

    element_to_graph = {}
    for k, v in subg_node.items():
        element_to_graph[k] = v
    
    for k, v in subg_edge.items():
        element_to_graph[k] = v

    return element_to_graph           #, [subg_node, subg_edge]


def get_operator(element, qubit_num):
    op = Observable(qubit_num)
    if isinstance(element, int):
        op.add_operator(1.0, "Z " + str(element))
    else:
        op.add_operator(1.0, "Z " + str(element[0]) + " Z " + str(element[1]))
    return op


def get_QAOA_circuit(graph, params, node_to_qubit):
    global p, nodes_weight, edges_weight
    gamma, beta = params[:p], params[p:]

    circ =QuantumCircuit(len(graph))

    for k in range(p):
        for nd in graph.nodes:
            u = node_to_qubit[nd]
            if k == 0:
                circ.add_H_gate(u)
            circ.add_RZ_gate(u, 2 * gamma[k] * nodes_weight[nd])

        for ed in graph.edges:
            u, v = node_to_qubit[ed[0]], node_to_qubit[ed[1]]
            if u == v:
                continue
            circ.add_CNOT_gate(u, v)
            circ.add_RZ_gate(v, 2 * gamma[k] * edges_weight[ed[0], ed[1]])
            circ.add_CNOT_gate(u, v)

        for nd in graph.nodes:
            u = node_to_qubit[nd]
            circ.add_RX_gate(u, 2 * beta[k])

    return circ
        

def get_expectation(ele_graph_item, params):
    global nodes_weight, edges_weight, use_gpu

    original_e, graph = ele_graph_item
    node_list = list(graph.nodes)
    node_to_qubit = defaultdict(int)
    for i in range(len(node_list)):
        node_to_qubit[node_list[i]] = i

    circ = get_QAOA_circuit(graph, params, node_to_qubit)
    if use_gpu is False:
        state = QuantumState(len(graph))
    else:
        state = QuantumStateGpu(len(graph))
    state.set_zero_state()
    circ.update_quantum_state(state)

    if isinstance(original_e, int):
        weight = nodes_weight[original_e]
        op = get_operator(node_to_qubit[original_e], len(node_list))
    else:
        weight = edges_weight[original_e]
        op = get_operator((node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]), len(node_list))

    exp_res = op.get_expectation_value(state)
    # print("expectation calculated by statistics is:", exp_res)
    return weight, exp_res


def expectation_calculation_serial(params):
    global element_to_graph
    # print("element to graph is: ", element_to_graph)
    exp_res = 0
    for item in element_to_graph.items():
        w_i, exp_i = get_expectation(item, params)
        exp_res += w_i * exp_i
    
    exp_path.append(exp_res)
    print("Total expectation of original graph is:", exp_res)
    return exp_res


def run(graph, params):
    global exp_path, element_to_graph, nodes_weight, edges_weight
    exp_path = []
    nodes_weight, edges_weight = get_graph_weights(graph)
    element_to_graph = graph_decompose(graph)
    res = minimize(expectation_calculation_serial, np.asarray(params), method='COBYLA', tol=1e-14,
                    jac=rosen_der, options={'gtol': 1e-8, 'disp': True})   #s'maxiter': 30, 

    return res


def optimization_visualization(exp_path):
    plt.figure()
    plt.plot(range(1, len(exp_path) + 1), exp_path, "ob-", label="qulacs")
    plt.ylabel("Expectation value")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()


def sampling_visualization(counts, state_num):
    state_counts = [counts[i] for i in range(state_num)]
    plt.figure()
    plt.bar(range(state_num), state_counts)
    plt.ylabel("Count number")
    plt.xlabel("State")
    plt.legend()
    plt.show()


def get_result_counts(graph, params):  # 直接将最优参数值带入整个大电路中运行qubit将只能到18
    global use_gpu

    node_list = list(graph.nodes)
    node_to_qubit = defaultdict(int)
    for i in range(len(node_list)):
        node_to_qubit[node_list[i]] = i
    circ = get_QAOA_circuit(graph, params, node_to_qubit)
    if use_gpu is False:
        state = QuantumState(len(graph))
    else:
        state = QuantumStateGpu(len(graph))
    state.set_zero_state()
    circ.update_quantum_state(state)
    state_samplings = state.sampling(1024)

    counts = defaultdict(int)
    for i in state_samplings:
        counts[i] += 1
    return counts


if __name__ == "__main__":

    p_list = [3, 5, 7, 9]
    node_num_list = [10, 50, 100, 500, 1000]  # , 500, 1000
    time_cpu, time_gpu = [], []
    res_cpu, res_gpu = [], []

    for p in p_list:
        for node_num in node_num_list:
            g = nx.Graph()
            for i in range(node_num):
                g.add_node(i, weight=0)
                if i > 0:
                    g.add_edge(i, i - 1, weight=-1)

            params = [random.randint(0, 1) for i in range(2 * p)]

            use_gpu = False
            st = time.time()
            res = run(g, params)
            ed = time.time()
            time_cpu.append(ed - st)
            res_cpu.append(res.fun)

            use_gpu = True
            st = time.time()
            res = run(g, params)
            ed = time.time()
            time_gpu.append(ed - st)
            res_gpu.append(res.fun)

        plt.figure(1)
        plt.plot(node_num_list, time_cpu, "ob-", label="CPU")
        plt.plot(node_num_list, time_gpu, "^r-", label="GPU")
        plt.ylabel("Time Cost(s)")
        plt.xlabel("node number")
        plt.title("P is %d" % p)
        plt.legend()
        plt.savefig('/home/puyanan/Qcover_GPU/result_log/qulacs/time_cost_%d.png' % p)
        # plt.show()

        plt.figure(2)
        plt.plot(node_num_list, res_cpu, "*g-", label="CPU")
        plt.plot(node_num_list, res_gpu, "dy-", label="GPU")
        plt.ylabel("Expectation")
        plt.xlabel("node number")
        plt.title("P is %d" % p)
        plt.legend()
        plt.savefig('/home/puyanan/Qcover_GPU/result_log/qulacs/expectation_res_%d.png' % p)
        # plt.show()
        plt.close('all')

    # test one example
    # p = 10
    # g = nx.Graph()
    # node_num = 5
    # for i in range(node_num):
    #     g.add_node(i, weight=0)
    #     if i > 0:
    #         g.add_edge(i, i - 1, weight=-1)
    # params = [random.randint(0, 1) for i in range(2*p)]
    # res = run(g, params)
    # print("Expectation of the Hamitanion is: ", res)
    # counts = get_result_counts(g, res.x)
    # sampling_visualization(counts, pow(2, len(g)))
    # print("element to graph is: ", element_to_graph)


    # nodes = [(0, 0), (1, 0), (2, 0)]
    # edges = [(0, 1, -1), (1, 2, -1)]
    # for nd in nodes:
    #     u, w = nd[0], nd[1]
    #     g.add_node(int(u), weight=int(w))
    # for ed in edges:
    #     u, v, w = ed[0], ed[1], ed[2]
    #     g.add_edge(int(u), int(v), weight=int(w))