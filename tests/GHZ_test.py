import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy import optimize as opt
from collections import defaultdict
import matplotlib.pyplot as plt
import random

from qiskit import *
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.aqua.operators import X, Y, Z, I


def get_GHZ_circuit(p, graph, params):
    # nodew = nx.get_node_attributes(graph, 'weight')
    edges_weight = nx.get_edge_attributes(graph, 'weight')

    gamma_list, beta_list = params[: p], params[p:]
    circ_qreg = QuantumRegister(len(graph), 'q')
    circ = QuantumCircuit(circ_qreg)

    pivot = len(graph) // 2

    for k in range(p):
        for nd in graph.nodes:
            if k == 0:
                circ.h(nd)

        for edge in graph.edges:
                u, v = edge[0], edge[1]
                if u == v:
                    continue
                circ.rzz(2 * gamma_list[k] * edges_weight[edge[0], edge[1]], u, v)  #

        circ.barrier()
        for nd in graph.nodes:
            if nd != pivot:
                circ.rx(2 * beta_list[k], nd)
                
        # circ.barrier()

    # circ.measure_all()
    return circ


def get_operator(element, qubit_num):
    op = 1
    for i in range(qubit_num):
        if i in list(element):
            op = np.kron(Z.to_matrix(), op)
        else:
            op = np.kron(I.to_matrix(), op)
    return op


def get_expectation(element_graph, p):

    original_e, graph = element_graph
    op = get_operator(original_e, len(graph))
    circ = get_GHZ_circuit(p, graph, params)
    # circ.measure_all()
    # print(op)
    # circ.draw(output='mpl', interactive=True)
    circ.save_statevector()

    sim = Aer.get_backend('qasm_simulator')  
    result = sim.run(circ, seed_simulator=43, nshots=1024).result()
    out_state = result.get_statevector()
    exp_res = np.matmul(np.matmul(out_state.conj().T, op), out_state).real

    ndw = nx.get_node_attributes(graph, 'weight')
    edw = nx.get_edge_attributes(graph, 'weight')
    if isinstance(original_e, int):
        weight = ndw[original_e]
    else:
        weight = edw[original_e]

    return weight, exp_res


def expectation_calculation(element_to_graph, p):
    res = 0
    # st = time.time()
    for item in element_to_graph.items():
        w_i, exp_i = get_expectation(item, p)
        res += w_i * exp_i  #
    # ed = time.time()
    # print("exp one cost:", ed - st)
    # print("Total expectation of original graph is: ", res)
    return res


def get_expectation_statistics(params, p, graph):
    circ = get_GHZ_circuit(p, graph, params)
    circ.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    counts = backend.run(circ, seed_simulator=43, nshots=1024).result().get_counts()

    def Ising_obj(x, G):
        obj = 0
        for i, j in G.edges():
            if x[i] == x[j]:
                obj -= 1
            else:
                obj += 1
        return obj

    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = Ising_obj(bitstring, graph)
        avg += obj * count
        sum_count += count
    exp_res = avg / sum_count
    
    print("expectation calculated by statistics is:", exp_res)
    return exp_res


def get_result_counts(p, params, graph):  # 直接将最优参数值带入整个大电路中运行qubit将只能到18

    params = params.flatten()
    circ = get_GHZ_circuit(p, graph, params)
    circ.measure_all()   # if use sample method, can't use measure
    circ.draw(output='mpl', interactive=True)

    sim = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(circ, sim)
    job_sim = sim.run(transpiled_qc, shots=1024)
    hist = job_sim.result().get_counts(transpiled_qc)

    for key, val in hist.items():
        print("%s has %s" % (key, str(val)))

    from qiskit.visualization import plot_histogram
    plot_histogram(hist)
    plt.show()
    return hist


if __name__ == '__main__':
    p = 1
    node_num = 5

    # build graph
    g = nx.Graph()
    g.add_node(0, weight=0)
    for i in range(1, node_num):
        g.add_node(i, weight=0)
        g.add_edge(i - 1, i, weight=-1)

    # nx.draw(g)
    # params = [0.463 for i in range(2 * p)]
    # params = [2.63560454, 0.39839157]
    params = [random.random() for i in range(2 * p)]
    from Qcover.core import Qcover
    from Qcover.optimizers import COBYLA, Interp
    from Qcover.backends import CircuitByQulacs #, CircuitByQiskit
    from Qcover.backends.circuitbyqiskit_statistic import CircuitByQiskit
    qulacs_bc = CircuitByQulacs(research="GHZ")
    qiskit_bc = CircuitByQiskit(research="GHZ", expectation_calc_method="statistic")  #"sample" "statevector"

    optc = COBYLA()  #options={'tol': 1e-8, 'disp': True}
    opti = Interp(optimize_method="COBYLA", options={'tol': 1e-8, 'disp': False})
    qc = Qcover(g, p,
                # research_obj="QAOA",
                optimizer=optc,  # @ optc,
                backend=qiskit_bc)  # qulacs_bc, qt, , cirq_bc, projectq_bc

    import time
    st = time.time()
    sol = qc.run(is_parallel=False, mode='QAQA')  #GHZ True
    ed = time.time()
    print("time cost by QAOA is:", ed - st)
    print("solution is:", sol)
    params = sol["Optimal parameter value"]
    qc.backend._pargs = params
    out_count = qc.backend.get_result_counts(params)
    res_exp = qc.backend.expectation_calculation()
    print("the optimal expectation is: ", res_exp)
    qc.backend.sampling_visualization(out_count)

    # test the expectation of whole graph with statistics method
    # circ_g = get_GHZ_circuit(p, g, params)
    # circ_g.measure_all()
    # circ_g.draw(output='mpl', interactive=True)
    # res_stc = minimize(get_expectation_statistics,
    #                      x0=np.array(params),
    #                      args=(p, g),
    #                      method='COBYLA',
    #                      jac=opt.rosen_der)
    # res_stc = get_expectation_statistics(circ_g, g)
    # print("expectation calculated by statistics is: ", res_stc)  # 1.650390625
    # res = minimize(get_expectation_statistics, theta, method="COBYLA")

    # test the expectation of whole graph with statevector method
    # element_to_graph = {}
    # for ed in g.edges():
    #     element_to_graph[ed] = g
    # res_sta = expectation_calculation(element_to_graph, p)
    # print("expectation calculated by statevector is: ", res_sta)