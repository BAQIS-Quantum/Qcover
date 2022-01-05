import os
import time
import warnings
import itertools

from collections import defaultdict, Callable
import numpy as np
import sympy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import networkx as nx

from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import RX, RZ, CNOT, merge


class CircuitByQulacs:
    """generate a instance of CircuitByQulacs"""

    def __init__(self,
                 # p: int = 1,
                 nodes_weight: list = None,
                 edges_weight: list = None,
                 is_parallel: bool = None) -> None:
        """initialize a instance of CircuitByCirq"""

        self._p = None
        self._nodes_weight = nodes_weight
        self._edges_weight = edges_weight
        self._is_parallel = False if is_parallel is None else is_parallel

        self._element_to_graph = None
        self._pargs = None
        self._expectation_path = []

    @staticmethod
    def get_operator(element, qubit_num):
        op = Observable(qubit_num)
        if isinstance(element, int):
            op.add_operator(1.0, "Z " + str(element))
        else:
            op.add_operator(1.0, "Z " + str(element[0]) + " Z " + str(element[1]))

        return op

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

        node_to_qubit = defaultdict(int)
        node_list = list(graph.nodes)
        for i in range(len(node_list)):
            node_to_qubit[node_list[i]] = i

        state = QuantumState(len(node_list))
        state.set_zero_state()

        circ = QuantumCircuit(len(node_list))
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

        if isinstance(original_e, int):
            weight = self._nodes_weight[original_e]
            op = self.get_operator(node_to_qubit[original_e], len(node_list))
        else:
            weight = self._edges_weight[original_e]
            op = self.get_operator((node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]), len(node_list))

        circ.update_quantum_state(state)
        exp_res = op.get_expectation_value(state)

        return weight * exp_res

    def expectation_calculation(self, p=None):
        if self._is_parallel:
            return self.expectation_calculation_parallel(p)
        else:
            return self.expectation_calculation_serial(p)

    def expectation_calculation_serial(self, p=None):
        res = 0
        for item in self._element_to_graph.items():
            res += self.get_expectation(item, p)

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

        res = sum(circ_res[0])
        print("Total expectation of original graph is: ", res)
        self._expectation_path.append(res)
        return res

    def visualization(self):
        plt.figure()
        plt.plot(range(1, len(self._expectation_path) + 1), self._expectation_path, "ob-", label="qulacs")
        plt.ylabel('Expectation value')
        plt.xlabel('Number of iterations')
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