import os
import time
import itertools

from collections import defaultdict, Callable
import numpy as np
import sympy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import networkx as nx

from Qcover.simulator import Qcircuit, Qcodes
from Qcover.backends import Backend
from Qcover.utils import get_graph_weights
import warnings
warnings.filterwarnings("ignore")


class CircuitByQton(Backend):
    """generate a instance of CircuitByQton"""

    def __init__(self,
                 research: str = "QAOA",
                 expectation_calc_method: str = "statevector",
                 is_parallel: bool = None) -> None:
        """initialize a instance of CircuitByCirq"""
        super(CircuitByQton, self).__init__()

        self._p = None
        self._origin_graph = None
        self._is_parallel = False if is_parallel is None else is_parallel
        self._expectation_calc_method = expectation_calc_method
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
    def get_operator(self):
        pass

    def get_QAOA_circuit(self, p, graph, node_to_qubit):
        circ = Qcircuit(len(graph))
        circ.mode = self._expectation_calc_method
        gamma_list, beta_list = self._pargs[: p], self._pargs[p:]
        for k in range(p):
            for i in graph.nodes:
                u = node_to_qubit[i]
                if k == 0:
                    circ.h(u)
                circ.rz(u, 2 * gamma_list[k] * self._nodes_weight[i])

            for edge in graph.edges:
                u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
                if u == v:
                    continue
                circ.rzz(u, v, 2 * gamma_list[k] * self._edges_weight[edge[0], edge[1]])

            for nd in graph.nodes:
                u = node_to_qubit[nd]
                circ.rx(u, 2 * beta_list[k])

        return circ

    def get_GHZ_circuit(self, p, graph, node_to_qubit):
        circ = Qcircuit(len(graph))
        circ.mode = self._expectation_calc_method
        gamma_list, beta_list = self._pargs[: p], self._pargs[p:]

        pivot = len(self._origin_graph) // 2
        for k in range(p):
            for nd in graph.nodes:
                u = node_to_qubit[nd]
                if k == 0:
                    circ.h(u)

            for edge in graph.edges:
                u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
                if u == v:
                    continue
                circ.rzz(u, v, 2 * gamma_list[k] * self._edges_weight[edge[0], edge[1]])

            for nd in graph.nodes:
                if nd != pivot:
                    u = node_to_qubit[nd]
                    circ.rx(u, 2 * beta_list[k])

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

        node_to_qubit = defaultdict(int)
        node_list = list(graph.nodes)
        for i in range(len(node_list)):
            node_to_qubit[node_list[i]] = i

        if self._research == "QAOA":
            circ = self.get_QAOA_circuit(p, graph, node_to_qubit)
        elif self._research == "GHZ":
            pivot = len(self._origin_graph) // 2
            if pivot in graph:
                circ = self.get_GHZ_circuit(p, graph, node_to_qubit)
            else:
                circ = self.get_QAOA_circuit(p, graph, node_to_qubit)

        state_circ = circ.state.copy()
        if isinstance(original_e, int):
            weight = self._nodes_weight[original_e]
            circ.z(node_to_qubit[original_e])
            # op = self.get_operator(node_to_qubit[original_e], len(node_list))
        else:
            weight = self._edges_weight[original_e]
            circ.z(node_to_qubit[original_e[0]])
            circ.z(node_to_qubit[original_e[1]])
            # op = self.get_operator((node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]), len(node_list))

        state_opc = circ.state.copy()
        exp_res = np.inner(state_opc.conj(), state_circ)

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

    def optimization_visualization(self):
        plt.figure()
        plt.plot(range(1, len(self._expectation_path) + 1), self._expectation_path, "ob-", label="qulacs")
        plt.ylabel('Expectation value')
        plt.xlabel('Number of iterations')
        plt.legend()
        plt.show()

    # def get_result_counts(self, params, graph):  # 直接将最优参数值带入整个大电路中运行qubit将只能到18
    #
    #     params = params.flatten()
    #     _, circ = self.graph_to_circuit(params, graph)
    #     circ.draw(output='mpl', interactive=True)
    #     circ.measure_all()   # if use sample method, can't use measure
    #
    #     sim = Aer.get_backend('qasm_simulator')
    #     transpiled_qc = transpile(circ, sim)
    #     job_sim = sim.run(transpiled_qc, shots=1024)
    #     hist = job_sim.result().get_counts(transpiled_qc)
    #     for key, val in hist.items():
    #         print("%s has %s" % (key, str(val)))
    #
    #     from qiskit.visualization import plot_histogram
    #     plot_histogram(hist)
    #     plt.show()
    #     return hist
