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
                 expectation_calc_method: str = "statevector", # or sample
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

    def get_operator(self, element, qubit_num):
        if self._expectation_calc_method == "statevector":
            op = 1
            for i in range(qubit_num):
                if i in list(element):
                    op = np.kron(Z.to_matrix(), op)
                else:
                    op = np.kron(I.to_matrix(), op)
        else:
            op = Z if 0 in list(element) else I
            for i in range(1, qubit_num):
                if i in list(element):
                    op = Z ^ op
                else:
                    op = I ^ op
        return op

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

        # circ.measure_all()
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

            # circ.measure_all()
            return circ

    def get_expectation(self, element_graph, p=None):
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
        # circ.draw(output='mpl', interactive=True)
        if isinstance(original_e, int):
            weight = self._nodes_weight[original_e]
            op = self.get_operator([node_to_qubit[original_e]], len(node_list))
        else:
            weight = self._edges_weight[original_e]
            op = self.get_operator([node_to_qubit[original_e[0]], node_to_qubit[original_e[1]]], len(node_list))

        if self._expectation_calc_method == "statevector":
            circ.save_statevector()
            sim = Aer.get_backend('qasm_simulator')  #aer
            result = sim.run(circ, seed_simulator=47, nshots=102400).result()
            out_state = result.get_statevector()
            exp_res = np.matmul(np.matmul(out_state.conj().T, op), out_state).real

        elif self._expectation_calc_method == "sample":
            subc = CircuitStateFn(circ)
            backend = Aer.get_backend('qasm_simulator')
            q_instance = QuantumInstance(backend, shots=1024)
            measurable_expression = StateFn(op, is_measurement=True).compose(subc)
            expectation = PauliExpectation().convert(measurable_expression)
            sampler = CircuitSampler(q_instance).convert(expectation)
            exp_res = sampler.eval().real

        elif self._expectation_calc_method == "statistic":
            circ.measure_all()
            backend = Aer.get_backend('qasm_simulator')
            backend.shots = 1024
            counts = backend.run(circ, seed_simulator=10, nshots=1024 * 100).result().get_counts()
            # counts = backend.run(circ, seed_simulator=43, nshots=1024).result().get_counts()

            def Ising_obj(x, G):
                obj = 0
                for ed in G.edges():
                    if x[node_to_qubit[ed[0]]] == x[node_to_qubit[ed[1]]]:
                        obj -= self._edges_weight[ed]
                    else:
                        obj += self._edges_weight[ed]
                return obj

            avg = 0
            sum_count = 0
            for bitstring, count in counts.items():
                obj = Ising_obj(bitstring, graph)
                avg += obj * count
                sum_count += count
            exp_res = avg / sum_count

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
        cpu_num = cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

        res = 0
        # st = time.time()
        for item in self._element_to_graph.items():
            w_i, exp_i = self.get_expectation(item, p)
            if isinstance(item[0], tuple):  #origin format of RQAOA
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
        circ.measure_all()   # if use sample method, can't use measure

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
