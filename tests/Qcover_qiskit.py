# -*- coding: utf-8 -*-
"""
Created on Wed Jul  12 10:12:33 2021

@author: YaNan Pu
"""
from cProfile import label
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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble, transpile
from qiskit.utils import QuantumInstance
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from sympy import false

global p, backend, element_to_graph, exp_path, nodes_weight, edges_weight 

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
                new_nodes = {(nd2, nodew[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}  #graph.adj[nd1]
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


def graph_decompose(graph):

    subg_node = generate_subgraph(graph, "node")    #, subg_setv
    subg_edge = generate_subgraph(graph, "edge")    #, subg_sete

    element_to_graph = {}
    for k, v in subg_node.items():
        element_to_graph[k] = v
    
    for k, v in subg_edge.items():
        element_to_graph[k] = v

    return element_to_graph           #, [subg_node, subg_edge]


def get_QAOA_circuit(graph, params, node_to_qubit):
    global p, nodes_weight, edges_weight
    gamma, beta = params[:p], params[p:]
    circ_qreg = QuantumRegister(len(graph), 'q')
    circ =QuantumCircuit(circ_qreg)

    for k in range(p):
        for nd in graph.nodes:
            u = node_to_qubit[nd]
            if k == 0:
                circ.h(u)
            circ.rz(2 * gamma[k] * nodes_weight[nd], u)

        circ.barrier()
        for ed in graph.edges:
            u, v = node_to_qubit[ed[0]], node_to_qubit[ed[1]]
            if u == v:
                continue
            circ.rzz(2 * gamma[k] * edges_weight[ed[0], ed[1]], u, v)

        circ.barrier()
        for nd in graph.nodes:
            circ.rx(2 * beta[k], node_to_qubit[nd])

    circ.measure_all()
    return circ
        

def get_expectation_statistics(ele_graph_item, params):
    global nodes_weight, edges_weight, backend

    original_e, graph = ele_graph_item
    node_list = list(graph.nodes)
    node_to_qubit = defaultdict(int)
    for i in range(len(node_list)):
        node_to_qubit[node_list[i]] = i

    circ = get_QAOA_circuit(graph, params, node_to_qubit)

    # backend = Aer.get_backend('qasm_simulator')
    # backend.shots = 1024
    counts = backend.run(circ, seed_simulator=10, nshots=1024 * 100).result().get_counts()

    def Ising_obj(x, g):
        obj = 0
        for ed in g.edges:
            u, v = node_to_qubit[ed[0]], node_to_qubit[ed[1]]
            if x[u] == x[v]:
                obj -= edges_weight[ed]
            else:
                obj += edges_weight[ed]

        for nd in g.nodes:
            u = node_to_qubit[nd]
            wx = 2 * int(x[u]) - 1
            obj += nodes_weight[u] * wx
        return obj

    svg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = Ising_obj(bitstring, graph)
        svg += obj * count
        sum_count += count
    exp_res = svg / sum_count

    if isinstance(original_e, int):
        weight = nodes_weight[original_e]
    else:
        weight = edges_weight[original_e]

    # print("expectation calculated by statistics is:", exp_res)
    return weight, exp_res


def expectation_calculation_serial(params):
    global element_to_graph
    # print("element to graph is: ", element_to_graph)
    exp_res = 0
    for item in element_to_graph.items():
        w_i, exp_i = get_expectation_statistics(item, params)
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


def get_result_counts(graph, params):  # 直接将最优参数值带入整个大电路中运行qubit将只能到18
    from qiskit.providers.aer import QasmSimulator
    params = params.flatten()
    node_list = list(graph.nodes)
    node_to_qubit = defaultdict(int)
    for i in range(len(node_list)):
        node_to_qubit[node_list[i]] = i
    circ = get_QAOA_circuit(graph, params, node_to_qubit)

    sim = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(circ, sim)
    job_sim = sim.run(transpiled_qc, shots=1024)
    hist = job_sim.result().get_counts(transpiled_qc)
    for key, val in hist.items():
        print("%s has %s" % (key, str(val)))

    return hist


def optimization_visualization(exp_path):
    plt.figure()
    plt.plot(range(1, len(exp_path) + 1), exp_path, "ob-", label="qiskit")
    plt.ylabel("Expectation value")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()


def sampling_visualization(hist):
    from qiskit.visualization import plot_histogram
    plot_histogram(hist)
    plt.show()



# def expectation_calculation_parallel(subg_to_circuit):
#     circ_res = []
#     res = 0

#     pool = Pool(os.cpu_count())
#     for subc in subg_to_circuit.values():
#         exps = pool.apply_async(func=get_statevector_expectation, args=(subc,))
#         res += exps.get()

#     pool.terminate()
#     pool.join()
#     print("Total expectation_statevector of original graph is: ", res)
#     # exp_sta_ip.append(res)
#     return res

# def expectation_calculation_parallel(subc):
#     res_expectation = 0
#     wires = [subc[0]] if isinstance(subc[0], int) else list(subc[0])
#     res_expectation = subc[1] * get_statevector_expectation(subc[2], *wires)

#     return res_expectation

# def get_result_statevector(self, params):
#     params = params.flatten()
#     _, circ = self.graph_to_circuit(params, self._simple_graph)
#     circ.save_statevector()
#     # subc.draw(output='mpl', interactive=True)

#     sim = Aer.get_backend('statevector_simulator')
#     qobj = assemble(circ)
#     result = sim.run(qobj).result()
#     out_state = result.get_statevector()
#     return out_state



if __name__ == "__main__":
    
    p_list = [3, 5, 7]
    node_num_list = [10, 50, 100, 500, 1000]  #, 500, 1000
    time_cpu, time_gpu = [], []
    res_cpu, res_gpu = [], []

    for p in p_list:
        for node_num in node_num_list:
            g = nx.Graph()
            for i in range(node_num):
                g.add_node(i, weight=0)
                if i > 0:
                    g.add_edge(i, i - 1, weight=-1)

            params = [random.randint(0, 1) for i in range(2*p)]

            backend = Aer.get_backend('qasm_simulator')
            backend.shots = 1024
            st = time.time()
            res = run(g, params)
            ed = time.time()
            time_cpu.append(ed - st)
            res_cpu.append(res.fun)

            backend = Aer.get_backend('aer_simulator')
            backend.shots = 1024
            backend.set_options(device='GPU', method="statevector")
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
        plt.savefig('/home/puyanan/Qcover_GPU/result_log/time_cost_%d.png' % p)
        # plt.show()

        plt.figure(2)
        plt.plot(node_num_list, res_cpu, "*g-", label="CPU")
        plt.plot(node_num_list, res_gpu, "dy-", label="GPU")
        plt.ylabel("Expectation")
        plt.xlabel("node number")
        plt.title("P is %d" % p)
        plt.legend()
        plt.savefig('/home/puyanan/Qcover_GPU/result_log/expectation_res_%d.png' % p)
        # plt.show()
        plt.close('all')



    # use_gpu = True
    # if use_gpu is True:
    #     backend = Aer.get_backend('aer_simulator')
    #     backend.shots = 1024
    #     backend.set_options(device='GPU',method="statevector")
    # else:
    #     backend = Aer.get_backend('qasm_simulator')
    #     backend.shots = 1024

    # params = [random.randint(0, 1) for i in range(2*p)]
    # st = time.time()
    # res = run(g, params)
    # ed = time.time()

    # print("time using is: ", ed - st)
    # print("Expectation of the Hamitanion is: ", res.fun)
    # visualization(exp_path)
    # counts = get_result_counts(g, res.x)
    # sampling_visualization(counts)
    # print("element to graph is: ", element_to_graph)
    # res_exp = expectation_calculation_serial(element_to_graph, params, p)




    # nodes = [(0, 0), (1, 0), (2, 0)]
    # edges = [(0, 1, -1), (1, 2, -1)]
    # for nd in nodes:
    #     u, w = nd[0], nd[1]
    #     g.add_node(int(u), weight=int(w))
    # for ed in edges:
    #     u, v, w = ed[0], ed[1], ed[2]
    #     g.add_edge(int(u), int(v), weight=int(w))