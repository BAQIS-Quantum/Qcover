import numpy as np
import networkx as nx
from typing import Optional
from collections import defaultdict

from Qcover.utils import get_graph_weights, generate_weighted_graph, generate_graph_data
from Qcover.optimizers import Optimizer, COBYLA
from Qcover.backends import Backend, CircuitByQiskit
from Qcover.core import Qcover
import warnings
warnings.filterwarnings("ignore")


class QAOA_Generate:
    def __init__(self,
                 graph: nx.Graph = None,
                 p: int = 1,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQiskit(),
                 ) -> None:

        assert graph is not None
        self._p = p
        self._original_graph = graph
        self._qc = Qcover(self._original_graph,
                          self._p,
                          optimizer=optimizer,
                          backend=backend,
                          research_obj="QAOA")

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, ap):
        self._p = ap

    @property
    def qc(self):
        return self._qc

    @property
    def original_graph(self):
        return self._original_graph

    @original_graph.setter
    def original_graph(self, graph):
        """
        according to the type of graph(nx.graph / tuple) to set the value of
        self._original_graph
        """
        if isinstance(graph, nx.Graph):
            self._original_graph = graph
        elif isinstance(graph, tuple):
            assert (len(graph) >= 2) and (len(graph) <= 3)

            if len(graph) == 2:
                node_num, edge_num = graph
                wr = None
            elif len(graph) == 3:
                node_num, edge_num, wr = graph

            nodes, edges = generate_graph_data(node_num, edge_num, wr)
            self._original_graph = generate_weighted_graph(nodes, edges)
        elif isinstance(graph, list):
            assert len(graph) == 3
            node_list, edge_list, weight_range = graph
            self._original_graph = generate_weighted_graph(node_list, edge_list, weight_range)
        else:
            print("Error: the argument graph should be a instance of nx.Graph "
                  "or a tuple formed as (node_num, edge_num)")

    def run(self, is_parallel=False):
        # nx.draw(self._g)
        # if self._qc is None:
        #     self._qc = Qcover(self._g,
        #                       self._p,
        #                       optimizer=self._optimizer,
        #                       backend=self._backend,
        #                       research_obj="GHZ")
        res = self._qc.run(is_parallel=is_parallel)  # True
        return res


if __name__ == '__main__':
    p = 3
    g = nx.Graph()
    nodes = [(0, 0), (1, 0), (2, 0)]
    edges = [(0, 1, 1), (1, 2, 1)]

    for nd in nodes:
        u, w = nd[0], nd[1]
        g.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
        g.add_edge(int(u), int(v), weight=int(w))

    from Qcover.optimizers import COBYLA, Interp, SLSQP, L_BFGS_B, GradientDescent, SPSA, Fourier
    optf = Fourier(p=p, q=4, r=3)
    optc = COBYLA(options={'maxiter': 300, 'disp': True, 'rhobeg': 1.0, 'tol': 1e-12}) #, initial_point=opti_params
    opti = Interp(optimize_method="COBYLA", options={'tol': 1e-12, 'disp': False}) #, initial_point=opti_params, "optimal_value": -2*node_num, "approximate_ratio": 0.9
    opts = SLSQP(options={'maxiter': 100,
                          'ftol': 1e-06,
                          'iprint': 1,
                          'disp': False,
                          'eps': 1.4901161193847656e-08,
                          'finite_diff_rel_step': None})

    optl = L_BFGS_B(options={'disp': None,
                             'maxcor': 10,
                             'ftol': 2.220446049250313e-09,
                             'gtol': 1e-05,
                             'eps': 1e-08,
                             'maxfun': 15000,
                             'maxiter': 15000,
                             'iprint': -1,
                             'maxls': 20,
                             'finite_diff_rel_step': None})  #

    optg = GradientDescent(options={'maxiter':300, 'learning_rate':0.05, 'tol':1e-6})
    opta = SPSA(options={'A':34, 'R': 15, 'maxiter':300, 'tol':1e-6})
    from Qcover.backends import CircuitByQiskit, CircuitByQton, CircuitByQulacs, CircuitByProjectq, CircuitByTensor, CircuitByCirq

    qiskit_bc = CircuitByQiskit(expectation_calc_method="statevector")
    qulacs_bc = CircuitByQulacs()

    # cirq_bc = CircuitByCirq()
    # projectq_bc = CircuitByProjectq()   # Bugs need to fix
    # ts = CircuitByTensor()
    # qt = CircuitByQton()  #expectation_calc_method="tensor"

    qaoa = QAOA_Generate(graph=g,
                           p=p,
                           optimizer=optc,
                           backend=qiskit_bc)  #qt, , cirq_bc, projectq_bc, qulacs_bc
    res = qaoa.run()

    print("solution is:", res)
    params = res["Optimal parameter value"]
    # params = [2.40756694, 2.37257408, 2.34787417, 0.79040416, 0.78994118,
    #    0.73654257]
    qaoa._qc.backend._pargs = params

    res_exp = qaoa._qc.backend.expectation_calculation()
    print("optimal parameter value:", qaoa._qc.backend._pargs)
    print("the optimal expectation is: ", res_exp)
    out_count = qaoa._qc.backend.get_result_counts(params)
    import matplotlib.pyplot as plt
    from qiskit.visualization import plot_histogram
    plot_histogram(out_count)
    plt.show()

