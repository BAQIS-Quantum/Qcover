# This code is part of Qcover.
#
# (C) Copyright BAQIS 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Object to solve QAOA problems

The QAOA problems can be represented as an Ising model, and then be transformed to a DAG.
The directed acyclic graph is decomposed by a specified p value, and these subgraphs then
be transformed as circuits and be executed on simulators, using optimizer to get the
optimal parameters of the original Ising model
"""

import sys
import time
from itertools import permutations

from typing import Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from Qcover.utils import get_graph_weights, generate_weighted_graph
from Qcover.optimizers import Optimizer, COBYLA
from Qcover.backends import Backend, CircuitByQiskit, CircuitByTensor
from Qcover.exceptions import GraphTypeError, UserConfigError
import warnings
warnings.filterwarnings("ignore")


class Qcover:
    """
    Qcover is a QAOA solver
    """
    # pylint: disable=invalid-name
    def __init__(self,
                 graph: nx.Graph = None,
                 p: int = 1,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQiskit(),
                 research_obj: str = "QAOA"
                 ) -> None:

        assert graph is not None
        self._simple_graph = graph
        self._p = p
        self._research_obj = research_obj
        self._backend = backend
        self._backend._p = p
        self._backend._origin_graph = graph
        self._backend._research = research_obj
        self._optimizer = optimizer
        self._optimizer._p = p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p):
        self._p = new_p

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, new_backend):
        self._backend = new_backend

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    # def generate_subgraph(self, dtype: str):
    def graph_decomposition(self, p):
        """
        according to the arguments of dtype and p to generate subgraphs from graph
        Args:
            graph (nx.Graph): graph to be decomposed
            dtype (string): set "node" or "edge", the ways according to which to decompose the graph
            p (int): the p of subgraphs
        Return:
            subg_dict (dict) form as {node_id : subg, ..., (node_id1, node_id2) : subg, ...}
        """
        # if dtype not in ["node", "edge"]:
        #     print("Error: wrong dtype, dtype should be node or edge")
        #     return None
        if p <= 0:
            warnings.warn(" the argument of p should be >= 1 in qaoa problem, "
                          "so p would be set to the default value at 1")
            p = 1

        nodes_weight, edges_weight = get_graph_weights(self._simple_graph)

        subg_dict = defaultdict(list)
        # if dtype == 'node':
        for node in self._simple_graph.nodes:
            node_set = {(node, nodes_weight[node])}
            edge_set = set()
            for i in range(p):
                new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in self._simple_graph[nd1[0]]}
                new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in self._simple_graph[nd1[0]]}
                node_set |= new_nodes
                edge_set |= new_edges

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[node] = subg
        # else:
        for edge in self._simple_graph.edges:
            node_set = {(edge[0], nodes_weight[edge[0]]), (edge[1], nodes_weight[edge[1]])}
            edge_set = {(edge[0], edge[1], edges_weight[edge[0], edge[1]])}

            for i in range(p):
                new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in self._simple_graph[nd1[0]]}
                new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in
                             self._simple_graph.adj[nd1[0]]}
                node_set |= new_nodes
                edge_set |= new_edges

            subg = generate_weighted_graph(node_set, edge_set)
            subg_dict[edge] = subg
        return subg_dict

    def calculate(self, pargs, p=None):
        """
        The framework function which use the backend to calculate the value of expectation,
        and be used as the object function in the optimization function of the optimizer
        Args:
            pargs: the value of the parameter alpha and beta in the circuit
            p: the integer used to define the number of layers the current circuit needs to be superimposed
        Returns:
            the value of expectation calculated by backends
        """
        p = self._p if p is None else p
        element_to_graph = self.graph_decomposition(p=p)

        # checking graph type of given problem
        if not isinstance(self._backend, CircuitByTensor):
            for k, v in element_to_graph.items():
                ncnt, ecnt = len(v.nodes), len(v.edges)
                try:
                    nreq1 = ncnt * (ncnt - 1) <= 2 * ecnt and ncnt >= 20
                    nreq2 = isinstance(k, int) and v.degree[k] >= 30
                    if nreq1 or nreq2:
                        raise GraphTypeError("The problem is transformed into a dense graph, " \
                           "which is difficult to be solved effectively by Qcover")
                except GraphTypeError as e:
                    print(e)
                    # if not self._hard_to_calcute:
                    #     print(e)
                    #     self._hard_to_calcute = True
                    # sys.exit()

        self._backend._pargs = pargs
        self._backend._element_to_graph = element_to_graph
        return self._backend.expectation_calculation(p)

    def run(self, is_parallel=False):
        self._backend._is_parallel = is_parallel
        x, fun, nfev = self._optimizer.optimize(objective_function=self.calculate)
        res = {"Optimal parameter value": x, "Expectation of Hamiltonian": fun, "Total iterations": nfev}
        return res

# usage example
if __name__ == '__main__':

    p = 1
    g = nx.Graph()
    nodes = [(0, 0), (1, 0), (2, 0)]
    edges = [(0, 1, 1), (1, 2, 1)]

    # nodes = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    # edges = [(0, 1, -1), (1, 2, -1), (2, 3, -1), (3, 4, -1)]

    for nd in nodes:
        u, w = nd[0], nd[1]
        g.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
        g.add_edge(int(u), int(v), weight=int(w))

    # node_num = 10
    # for nd in range(node_num):
    #     g.add_node(nd, weight=0)
    #     if nd < node_num - 1:
    #         g.add_edge(nd, nd + 1, weight=-1)

    #Ising test
    # g.add_node(0, weight=0)
    # for i in range(1, 30):
    #     g.add_node(i, weight=0)
    #     g.add_edge(i, i - 1, weight=-1)

    # nx.draw(g)
    # from Qcover.applications import MaxCut
    # mxt = MaxCut(g)
    # mxt = MaxCut(node_num=20, node_degree=3)
    #10 3 0.035  100 3 0.029
    #10 6 0.028  100 6 134.56/ce
    # g, shift = mxt.run()

    from Qcover.optimizers import GradientDescent, SPSA, Interp, SHGO

    optg = GradientDescent(options={'maxiter':300, 'learning_rate':0.05, 'tol':1e-6})
    opti = Interp(optimize_method="COBYLA", options={'tol': 1e-8, 'disp': False})
    opta = SPSA(options={'maxiter':300, 'tol':1e-6})
    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    opts = SHGO(options={'minimizer_kwargs':{'method':'SLSQP', 'options':{'ftol': 1e-12}}, 'sampling_method':'sobol'})
    qiskit_bc = CircuitByQiskit(expectation_calc_method="statevector")
    from Qcover.backends import CircuitByQton, CircuitByQulacs, CircuitByProjectq, CircuitByTensor, CircuitByCirq
    qulacs_bc = CircuitByQulacs()
    # cirq_bc = CircuitByCirq()
    # projectq_bc = CircuitByProjectq()
    # ts = CircuitByTensor()
    # qt = CircuitByQton()  #expectation_calc_method="tensor"

    qc = Qcover(g, p,
                # research_obj="QAOA",
                optimizer=optc,  #@ optc,
                backend=qulacs_bc)  #qiskit_bc, qt, , cirq_bc, projectq_bc

    st = time.time()
    sol = qc.run(is_parallel=False)  #True
    ed = time.time()
    print("time cost by QAOA is:", ed - st)
    print("solution is:", sol)
    params = sol["Optimal parameter value"]
    qc.backend._pargs = params
    out_count = qc.backend.get_result_counts(params)
    res_exp = qc.backend.expectation_calculation()
    print("the optimal expectation is: ", res_exp)
    qc.backend.sampling_visualization(out_count)

    # test RQAOA
    # import matplotlib.pyplot as plt
    # from Qcover.applications import MaxCut
    #
    # p = 1
    # node_num = [10, 50, 100, 500, 1000]
    # node_d = [3, 4, 5, 6]
    # for i in node_num:
    #     et, gt = [], []
    #     ev, gv = [], []
    #     for nd in node_d:
    #         print("node number is %d, node degree is %d" % (i, nd))
    #         print("----------------------------------------------")
    #         mxt = MaxCut(node_num=i, node_degree=nd)
    #         g, shift = mxt.run()
    #
    #         g_exp = g.copy()
    #         qc_exp = Qcover(g_exp, p,
    #                     optimizer=COBYLA(options={'tol': 1e-3, 'disp': True}),
    #                     backend=CircuitByQiskit(expectation_calc_method="statevector"))  # qt, ,qulacs_bc
    #
    #         st = time.time()
    #         sol_exp, sexp_exp = qc_exp.run(mode='RQAQA', node_threshold=1, iter_time=3, corr_method='expectation')
    #         ed = time.time()
    #         t1 = ed - st
    #
    #         g_g = g.copy()
    #         qc_g = Qcover(g_g, p,
    #                     optimizer=COBYLA(options={'tol': 1e-3, 'disp': True}),
    #                     backend=CircuitByQiskit(expectation_calc_method="statevector"))  # qt, ,qulacs_bc
    #         st = time.time()
    #         sol_g, sexp_g = qc_g.run(mode='RQAQA', node_threshold=1, iter_time=3, corr_method='g')
    #         ed = time.time()
    #         t2 = ed - st
    #
    #         exp_e, exp_g = 0, 0
    #         for (x, y) in g.nodes.data('weight', default=0):
    #             exp_e += y * (sol_exp[x] * 2 - 1)
    #             exp_g += y * (sol_g[x] * 2 - 1)
    #         for (u, v, c) in g.edges.data('weight', default=0):
    #             exp_e += c * (sol_exp[u] * 2 - 1) * (sol_exp[v] * 2 - 1)
    #             exp_g += c * (sol_g[u] * 2 - 1) * (sol_g[v] * 2 - 1)
    #
    #         et.append(t1)
    #         gt.append(t2)
    #
    #         ev.append(exp_e)
    #         gv.append(exp_g)
    #
    #         plt.figure(0)
    #         plt.plot(range(len(sexp_exp)), sexp_exp, "ob-", label="origin")
    #         plt.plot(range(len(sexp_g)), sexp_g, "^r-", label="correlation")
    #         plt.ylabel('expectation')
    #         plt.xlabel('iteration rounds')
    #         plt.title("node is %d, degree is %d" % (i, nd))
    #         plt.legend()
            # plt.savefig('/home/baqis/Qcover/result_log/correlation_exp/iter_%d_%d.png' % (i, nd))
            # plt.savefig('E:/Working_projects/QAOA/Qcover/result_log/correlation_exp/iter_%d_%d.png' % (i, nd))
        #     plt.savefig('/home/puyanan/Qcover/result_log/correlation_exp/iter_%d_%d.png' % (i, nd))
        #     plt.close('all')
        #
        # plt.figure(1)
        # plt.plot(node_d, et, "ob-", label="origin")
        # plt.plot(node_d, gt, "^r-", label="correlation")
        # plt.ylabel('Time cost')
        # plt.xlabel('node degree')
        # plt.title("node is %d" % i)
        # plt.legend()
        # plt.savefig('/home/baqis/Qcover/result_log/correlation_exp/time_cost_%d.png' % i)
        # plt.savefig('E:/Working_projects/QAOA/Qcover/result_log/correlation_exp/time_cost_%d.png' % i)
        # plt.savefig('/home/puyanan/Qcover/result_log/correlation_exp/time_cost_%d.png' % i)
        #
        # plt.figure(2)
        # plt.plot(node_d, ev, "*g-", label="origin")
        # plt.plot(node_d, gv, "dy-", label="correlation")
        # plt.ylabel('expectation of hamitonian')
        # plt.xlabel('node degree')
        # plt.title("node is %d" % i)
        # plt.legend()
        # plt.savefig('/home/baqis/Qcover/result_log/correlation_exp/hamitonian_exp_%d.png' % i)
        # plt.savefig('E:/Working_projects/QAOA/Qcover/result_log/correlation_exp/hamitonian_exp_%d.png' % i)
        # plt.savefig('/home/puyanan/Qcover/result_log/correlation_exp/hamitonian_exp_%d.png' % i)
        #
        # plt.close('all')




# print("time cost by RQAOA with expectation is:", t1)
# print("expectation list with expectation:", exp_e)
# print("solution of expectation is:", sol_exp)
# # {8: 0, 4: 1, 2: 0, 3: 1, 7: 0, 5: 1, 0: 0, 6: 0, 1: 1, 9: 1})
#
# print("time cost by RQAOA with correlation is:", t2)
# print("expectation list with correlation:", exp_g)
# print("solution of correlation is:", sol_g)




