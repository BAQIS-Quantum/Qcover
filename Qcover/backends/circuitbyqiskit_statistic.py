import itertools
import os
import time
from collections import defaultdict, Callable
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble, BasicAer, transpile
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn, CircuitOp, CircuitStateFn, \
    MatrixExpectation, X, Y, Z, I

from Qcover.utils import get_graph_weights
from Qcover.backends import Backend
import warnings
warnings.filterwarnings("ignore")


class CircuitByQiskit(Backend):
    """generate a instance of CircuitByQiskit"""
    def __init__(self,
                 expectation_calc_method: str = "statistic", # or sample
                 research: str = "QAOA",
                 is_parallel: bool = None) -> None:
        """initialize a instance of CircuitByQiskit"""
        super(CircuitByQiskit, self).__init__()
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

    def get_QAOA_circuit(self, p, graph, node_to_qubit):
        gamma_list, beta_list = self._pargs[: p], self._pargs[p:]
        circ_qreg = QuantumRegister(len(graph), 'q')
        circ = QuantumCircuit(circ_qreg)

        for k in range(p):
            for nd in graph.nodes:
                u = node_to_qubit[nd]
                if k == 0:
                    circ.h(u)
                circ.rz(2 * gamma_list[k] * self._nodes_weight[nd], u)

            for edge in graph.edges:
                u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
                if u == v:
                    continue
                circ.rzz(2 * gamma_list[k] * self._edges_weight[edge[0], edge[1]], u, v)

            for nd in graph.nodes:
                circ.rx(2 * beta_list[k], node_to_qubit[nd])

        circ.measure_all()
        return circ

    def get_GHZ_circuit(self, p, graph, node_to_qubit):
        gamma_list, beta_list = self._pargs[: p], self._pargs[p:]
        circ_qreg = QuantumRegister(len(graph), 'q')
        circ = QuantumCircuit(circ_qreg)

        pivot = len(self._origin_graph) // 2

        for k in range(p):
            for nd in graph.nodes:
                if k == 0:
                    circ.h(node_to_qubit[nd])

            for edge in graph.edges:
                    u, v = node_to_qubit[edge[0]], node_to_qubit[edge[1]]
                    if u == v:
                        continue
                    circ.rzz(2 * gamma_list[k] * self._edges_weight[edge[0], edge[1]], u, v)  #

            for nd in graph.nodes:
                if nd != pivot:
                    circ.rx(2 * beta_list[k], node_to_qubit[nd])

            circ.measure_all()
            return circ

    def create_qaoa_cir(self, G):
        N_qubits = len(G.nodes())
        p = len(self._pargs) // 2
        qc = QuantumCircuit(N_qubits)

        gamma = self._pargs[:p]
        beta = self._pargs[p:]

        for i in range((N_qubits - 1) // 2):
            qc.h(i)

        qc.ry(np.pi/2, (N_qubits - 1) // 2)

        for i in range((N_qubits + 1) // 2, N_qubits):
            qc.h(i)

        for irep in range(p):
            for pair in list(G.edges()):
                qc.rzz(-2 * gamma[irep], pair[0], pair[1])

            # for i in range(N_qubits):
            #   qc.rx(2*beta[irep], i)

            for i in range((N_qubits - 1) // 2):
                qc.rx(2 * beta[irep], i)
            for i in range((N_qubits + 1) // 2, N_qubits):
                qc.rx(2 * beta[irep], i)

        qc.measure_all()

        return qc

    def Ising_obj(self, x, G):
        obj = 0
        for i, j in G.edges():
            if x[i] == x[j]:
                obj -= 1
            else:
                obj += 1
        return obj

    def compute_expectation(self, counts, G):
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.Ising_obj(bitstring, G)
            avg += obj * count
            sum_count += count
        return avg / sum_count

    def get_expectation(self, G, p, shots=1024):
        p = self._p if p is None else p
        backend = Aer.get_backend('qasm_simulator')
        backend.shots = shots

        qc = self.create_qaoa_cir(G)
        counts = backend.run(qc, seed_simulator=10, nshots=1024 * 100).result().get_counts()

        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.Ising_obj(bitstring, G)
            avg += obj * count
            sum_count += count
        return avg / sum_count

    # def get_expectation(self, element_graph, p=None):
    #     if self._is_parallel is False:
    #         p = self._p if p is None else p
    #         original_e, graph = element_graph
    #     else:
    #         p = self._p if len(element_graph) == 1 else element_graph[1]
    #         original_e, graph = element_graph[0]
    #
    #     node_to_qubit = defaultdict(int)
    #     node_list = list(graph.nodes)
    #     for i in range(len(node_list)):
    #         node_to_qubit[node_list[i]] = i
    #
    #     if self._research == "QAOA":
    #         circ = self.get_QAOA_circuit(p, graph, node_to_qubit)
    #     elif self._research == "GHZ":
    #         pivot = len(self._origin_graph) // 2
    #         if pivot in graph:
    #             circ = self.get_GHZ_circuit(p, graph, node_to_qubit)
    #         else:
    #             circ = self.get_QAOA_circuit(p, graph, node_to_qubit)
    #     # circ.draw(output='mpl', interactive=True)
    #     if isinstance(original_e, int):
    #         weight = self._nodes_weight[original_e]
    #     else:
    #         weight = self._edges_weight[original_e]
    #
    #     backend = Aer.get_backend('qasm_simulator')
    #     counts = backend.run(circ, seed_simulator=43, nshots=1024).result().get_counts()
    #
    #     def Ising_obj(x, G):
    #         obj = 0
    #         for ed in G.edges():
    #             if x[node_to_qubit[ed[0]]] == x[node_to_qubit[ed[1]]]:
    #                 obj -= self._edges_weight[ed]
    #             else:
    #                 obj += self._edges_weight[ed]
    #         return obj
    #
    #     avg = 0
    #     sum_count = 0
    #     for bitstring, count in counts.items():
    #         obj = Ising_obj(bitstring, graph)
    #         avg += obj * count
    #         sum_count += count
    #     exp_res = avg / sum_count
    #
    #     return weight, exp_res

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
        cpu_num = cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

        res = 0
        # st = time.time()
        for item in self._element_to_graph.items():
            original_e, G = item
            w_i = self._nodes_weight[original_e]
            exp_i = self.get_expectation(G, p)
            # if isinstance(item[0], tuple):  #origin format of RQAOA
            self._element_expectation[item[0]] = exp_i
            res += w_i * exp_i  #
        # ed = time.time()
        # print("exp one cost:", ed - st)
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
        plt.plot(range(1, len(self._expectation_path) + 1), self._expectation_path, "ob-", label="Qiskit")
        plt.ylabel('Expectation value')
        plt.xlabel('Number of iterations')
        plt.legend()
        plt.show()

    # below is functions not used now
    def graph_to_circuit(self, params, graph, p=None, original_e=None):
        """
        transform the graph to circuit according to the computing_framework
        Args:
            graph (nx.Graph): graph to be transformed to circuit
            params (List): Optimal parameters   #np.array
            original_e (Optional[None, int, tuple])
        Return:
            if original_e=None, then the graph is the whole original graph generated by
            generate_weighted_graph(), so just return the circuit transformed by it

            if original_e is a int, then the subgraph is generated by node(idx = original_e
            in whole graph), so return the it's idx mapped by node_to_qubit[], and the circuit

            if original_e is a tuple, then the subgraph is generated by edge(node idx = original_e
            in whole graph), so return the it's idx mapped by node_to_qubit[] as
            tuple(mapped node_id1, mapped node_id2), and the circuit
        """

        # generate node_to_qubit[] map from node id in graph to [0, node_num in graph - 1]
        # so the subgraph would has it's own qubits number
        p = self._p if p is None else p

        node_to_qubit = defaultdict(int)
        node_list = list(graph.nodes)
        for i in range(len(node_list)):
            node_to_qubit[node_list[i]] = i

        self._pargs = params
        if self._research == "QAOA":
            circ = self.get_QAOA_circuit(p, graph, node_to_qubit)
        elif self._research == "GHZ":
            circ = self.get_GHZ_circuit(p, graph, node_to_qubit)

        if original_e is None:
            return original_e, circ
        elif isinstance(original_e, int):
            return node_to_qubit[original_e], circ
        else:
            return (node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]), circ

    def get_result_statevector(self, params, graph):

        params = params.flatten()
        _, circ = self.graph_to_circuit(params, graph)
        circ.save_statevector()
        # circ.draw(output='mpl', interactive=True)

        sim = Aer.get_backend('aer_simulator')
        result = sim.run(circ).result()
        out_state = result.get_statevector()

        return out_state

    def get_result_counts(self, params):  # 直接将最优参数值带入整个大电路中运行qubit将只能到18

        params = params.flatten()
        _, circ = self.graph_to_circuit(params, self._origin_graph)
        circ.draw(output='mpl', interactive=True)
        # circ.measure_all()   # if use sample method, can't use measure

        sim = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(circ, sim)
        job_sim = sim.run(transpiled_qc, shots=1024)
        hist = job_sim.result().get_counts(transpiled_qc)
        # for key, val in hist.items():
        #     print("%s has %s" % (key, str(val)))

        # from qiskit.visualization import plot_histogram
        # plot_histogram(hist)
        # plt.show()
        return hist

    def sampling_visualization(self, counts):
        from qiskit.visualization import plot_histogram
        plot_histogram(counts)
        plt.show()
