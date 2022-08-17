import os
import time
import itertools

from collections import defaultdict, Callable
import numpy as np
import sympy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import networkx as nx

from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import RX, RZ, CNOT, merge
from Qcover.backends import Backend
from Qcover.utils import get_graph_weights
import warnings
warnings.filterwarnings("ignore")


class CircuitByQulacs(Backend):
    """generate a instance of CircuitByQulacs"""

    def __init__(self,
                 research: str = "QAOA",
                 is_parallel: bool = None) -> None:
        """initialize a instance of CircuitByCirq"""
        super(CircuitByQulacs, self).__init__()

        self._p = None
        self._origin_graph = None
        self._is_parallel = False if is_parallel is None else is_parallel
        self._research = research

        self._nodes_weight = None
        self._edges_weight = None
        self._element_to_graph = None
        self._pargs = None
        self._expectation_path = []
        self._element_expectation = dict()

    @property
    def element_expectation(self):
        return self._element_expectation

    @staticmethod
    def get_operator(element, qubit_num):
        op = Observable(qubit_num)
        if isinstance(element, int):
            op.add_operator(1.0, "Z " + str(element))
        else:
            op.add_operator(1.0, "Z " + str(element[0]) + " Z " + str(element[1]))

        return op

    def get_QAOA_circuit(self, p, graph, node_to_qubit):  #, params=None
        circ = QuantumCircuit(len(graph))
        # if params is None:
        gamma_list, beta_list = self._pargs[: p], self._pargs[p:]
        for k in range(p):
            for i in graph.nodes:
                u = node_to_qubit[i]
                if k == 0:
                    circ.add_H_gate(u)
                circ.add_RZ_gate(u, 2 * gamma_list[k] * self._nodes_weight[i])

            for edge in graph.edges:
                u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
                if u == v:
                    continue
                circ.add_CNOT_gate(u, v)
                circ.add_RZ_gate(v, 2 * gamma_list[k] * self._edges_weight[edge[0], edge[1]])
                circ.add_CNOT_gate(u, v)

            for nd in graph.nodes:
                u = node_to_qubit[nd]
                circ.add_RX_gate(u, 2 * beta_list[k])

        return circ

    def get_GHZ_circuit(self, p, graph, node_to_qubit):
        circ = QuantumCircuit(len(graph))
        gamma_list, beta_list = self._pargs[: p], self._pargs[p:]

        pivot = len(self._origin_graph) // 2
        for k in range(p):
            for nd in graph.nodes:
                if k == 0:
                    circ.add_H_gate(node_to_qubit[nd])

            for edge in graph.edges:
                u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
                if u == v:
                    continue
                circ.add_CNOT_gate(u, v)
                circ.add_RZ_gate(v, 2 * gamma_list[k] * self._edges_weight[edge[0], edge[1]])
                circ.add_CNOT_gate(u, v)

            for nd in graph.nodes:
                if nd != pivot:
                    circ.add_RX_gate(node_to_qubit[nd], 2 * beta_list[k])

        return circ

    def get_expectation(self, element_graph, p=None):  #,
        """
        calculate the expectation of the subgraph
        Args:
            element_graph: tuple of (original node/edge, subgraph)

        Returns:
            expectation of the subgraph
        """
        if self._is_parallel is False:
            p = self._p if p is None else p
            original_e, graph = element_graph
        else:
            p = self._p if len(element_graph) == 1 else element_graph[1]
            original_e, graph = element_graph[0]

        node_list = list(graph.nodes)
        node_to_qubit = defaultdict(int)
        for i in range(len(node_list)):
            node_to_qubit[node_list[i]] = i

        state = QuantumState(len(graph.nodes))
        state.set_zero_state()

        if self._research == "QAOA":
            circ = self.get_QAOA_circuit(p, graph, node_to_qubit)
        elif self._research == "GHZ":
            pivot = len(self._origin_graph) // 2
            if pivot in graph:
                circ = self.get_GHZ_circuit(p, graph, node_to_qubit)
            else:
                circ = self.get_QAOA_circuit(p, graph, node_to_qubit)

        circ.update_quantum_state(state)

        if isinstance(original_e, int):
            weight = self._nodes_weight[original_e]
            op = self.get_operator(node_to_qubit[original_e], len(node_list))
        else:
            weight = self._edges_weight[original_e]
            op = self.get_operator((node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]), len(node_list))

        exp_res = op.get_expectation_value(state)

        return weight, exp_res

    def expectation_calculation(self, p=None):
        if self._nodes_weight is None or self._edges_weight is None:
            nodes_weight, edges_weight = get_graph_weights(self._origin_graph)
            self._nodes_weight, self._edges_weight = nodes_weight, edges_weight

        self._element_expectation = {}
        if self._is_parallel:
            return self.expectation_calculation_parallel(p)
        else:
            return self.expectation_calculation_serial(p)

    def expectation_calculation_serial(self, p=None):
        res = 0
        for item in self._element_to_graph.items():
            w_i, exp_i = self.get_expectation(item, p)
            if isinstance(item[0], tuple):
                self._element_expectation[item[0]] = exp_i
            res += w_i * exp_i

        print("Total expectation of original graph is: ", res)
        self._expectation_path.append(res)
        return res

    def expectation_calculation_parallel(self, p=None):
        cpu_num = 1
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

        circ_res = []
        args = list(itertools.product(self._element_to_graph.items(), [p]))
        pool = Pool(os.cpu_count())
        circ_res.append(pool.map(self.get_expectation, args))
        # circ_res.append(pool.map(self.get_expectation, list(self._element_to_graph.items()), chunksize=1))

        pool.terminate()  # pool.close()
        pool.join()
        res = 0
        for it in circ_res[0]:
            res += it[0] * it[1]
        # res = sum(circ_res[0])
        print("Total expectation of original graph is: ", res)
        self._expectation_path.append(res)
        return res

    def get_result_counts(self, params):
        node_list = list(self._origin_graph.nodes)
        node_to_qubit = defaultdict(int)
        for i in range(len(node_list)):
            node_to_qubit[node_list[i]] = i

        circ = self.get_QAOA_circuit(self._p, self._origin_graph, node_to_qubit)  #, params
        state = QuantumState(len(self._origin_graph))
        state.set_zero_state()
        circ.update_quantum_state(state)
        state_samplings = state.sampling(1024)

        counts = defaultdict(int)
        for i in state_samplings:
            counts[i] += 1

        return counts

    def optimization_visualization(self):
        plt.figure()
        plt.plot(range(1, len(self._expectation_path) + 1), self._expectation_path, "ob-", label="qulacs")
        plt.ylabel('Expectation value')
        plt.xlabel('Number of iterations')
        plt.legend()
        plt.show()

    def sampling_visualization(self, counts):
        state_num = pow(2, len(self._origin_graph))
        state_counts = [counts[i] for i in range(state_num)]
        plt.figure()
        plt.bar(range(state_num), state_counts)
        plt.ylabel("Count number")
        plt.xlabel("State")
        plt.legend()
        plt.show()



# test
# from qulacs import Observable, QuantumCircuit, QuantumState
# from qulacs.gate import Y,CNOT,merge
#
# state = QuantumState(3)
# state.set_Haar_random_state()
#
# circuit = QuantumCircuit(3)
# circuit.add_X_gate(0)
# merged_gate = merge(CNOT(0,1),Y(1))
# circuit.add_gate(merged_gate)
# circuit.add_RX_gate(1,0.5)
# circuit.update_quantum_state(state)
#
# observable = Observable(3)
# observable.add_operator(2.0, "X 2 Y 1 Z 0")
# observable.add_operator(-3.0, "Z 2")
# value = observable.get_expectation_value(state)
# print(value)