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
from Qcover.utils import get_graph_weights
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
    def p(self, ap):
        self._p = ap

    @property
    def backend(self):
        return self._backend

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def simple_graph(self):
        return self._simple_graph

    @simple_graph.setter
    def simple_graph(self, graph):
        """
        according to the type of graph(nx.graph / tuple) to set the value of
        self._simple_graph
        """
        if isinstance(graph, nx.Graph):
            self._simple_graph = graph
        elif isinstance(graph, tuple):
            assert (len(graph) >= 2) and (len(graph) <= 3)

            if len(graph) == 2:
                node_num, edge_num = graph
                wr = None
            elif len(graph) == 3:
                node_num, edge_num, wr = graph

            nodes, edges = Qcover.generate_graph_data(node_num, edge_num, wr)
            self._simple_graph = self.generate_weighted_graph(nodes, edges)
        elif isinstance(graph, list):
            assert len(graph) == 3
            node_list, edge_list, weight_range = graph
            self._simple_graph = Qcover.generate_weighted_graph(node_list, edge_list, weight_range)
        else:
            print(
                "Error: the argument graph should be a instance of nx.Graph or a tuple formed as (node_num, edge_num)")

    @staticmethod
    def generate_graph_data(node_num, edge_num, weight_range=10):
        """
        generate a simple graph‘s weights of nodes and edges with
        node number is node_num, edge number is edge_num
        Args:
            node_num (int): node number in graph
            edge_num (int): edge number in graph
            weight_range (int): weight range of nodes and edges
        Return:
            nodes(set of tuple(nid, node_weight)), edges(set of tuple(nid1, nid2, edge_weight))
        """
        if weight_range is None:
            weight_range = 10

        nodes = set()
        for i in range(node_num):
            ndw = np.random.choice(range(weight_range))
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
            for e in edges:  # without duplicated edges
                if set(e[:2]) == set([v, u]):
                    flg = 1
                    break
            if flg == 1:
                continue
            edw = np.random.choice(range(weight_range))
            edges |= {(u, v, edw)}
            cnt -= 1
        return nodes, edges

    @classmethod
    def generate_weighted_graph(cls, nodes, edges, weight_range=10):
        """
        generate graph from nodes list and edges list which identify the nodes and edges
        that should be add in graph, and the random weight range of every node and edge.
        Args:
            nodes (list/set): list of node idex / node-weight map, element form is tuple(nid, weight)
            edges (list/set): list of edge: (e_idex1, e_idex2) / edge-weight map, element form is tuple(nid1, nid2, edge_weight)
            weight_range (int): random weight range of every node and edge
        Returns:
            g (nx.Graph): graph generated by args
        """
        g = nx.Graph()
        if isinstance(nodes, list) and isinstance(edges, list):
            for v in nodes:
                w = np.random.choice(range(weight_range))
                g.add_node(v, weight=w)

            for e in edges:
                w = np.random.choice(range(weight_range))
                g.add_edge(e[0], e[1], weight=w)
        else:
            for item in nodes:
                g.add_node(item[0], weight=item[1])

            for item in edges:
                g.add_edge(item[0], item[1], weight=item[2])
        return g

    # @staticmethod
    # def get_graph_weights(graph):
    #     """
    #     get the weights of nodes and edges in graph
    #     Args:
    #         graph (nx.Graph): graph to get weight of nodes and edges
    #     Return:
    #         node weights form is dict{nid1: node_weight}, edges weights form is dict{(nid1, nid2): edge_weight}
    #     """
    #     nodew = nx.get_node_attributes(graph, 'weight')
    #     edw = nx.get_edge_attributes(graph, 'weight')
    #     edgew = edw.copy()
    #     for key, val in edw.items():
    #         edgew[(key[1], key[0])] = val
    #
    #     return nodew, edgew

    def generate_subgraph(self, graph, dtype: str, p):
        """
        according to the arguments of dtype and p to generate subgraphs from graph
        Args:
            graph (nx.Graph): graph to be decomposed
            dtype (string): set "node" or "edge", the ways according to which to decompose the graph
            p (int): the p of subgraphs
        Return:
            subg_dict (dict) form as {node_id : subg, ..., (node_id1, node_id2) : subg, ...}
        """
        if dtype not in ["node", "edge"]:
            print("Error: wrong dtype, dtype should be node or edge")
            return None

        nodes_weight, edges_weight = get_graph_weights(graph)

        subg_dict = defaultdict(list)
        if dtype == 'node':
            for node in graph.nodes:
                node_set = {(node, nodes_weight[node])}
                edge_set = set()
                for i in range(p):
                    new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                    new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                    node_set |= new_nodes
                    edge_set |= new_edges

                subg = self.generate_weighted_graph(node_set, edge_set)
                subg_dict[node] = subg
        else:
            for edge in graph.edges:
                node_set = {(edge[0], nodes_weight[edge[0]]), (edge[1], nodes_weight[edge[1]])}
                edge_set = {(edge[0], edge[1], edges_weight[edge[0], edge[1]])}

                for i in range(p):
                    new_nodes = {(nd2, nodes_weight[nd2]) for nd1 in node_set for nd2 in graph[nd1[0]]}
                    new_edges = {(nd1[0], nd2, edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in
                                 graph.adj[nd1[0]]}
                    node_set |= new_nodes
                    edge_set |= new_edges

                subg = self.generate_weighted_graph(node_set, edge_set)
                subg_dict[edge] = subg
        return subg_dict

    def graph_decomposition(self, graph, p):
        """
        according to dtype to decompose graph
        Args:
            graph (nx.Graph): graph to be composed
            p (int): the p of subgraphs
        """
        if p <= 0:
            warnings.warn(" the argument of p should be >= 1 in qaoa problem, "
                          "so p would be set to the default value at 1")
            p = 1

        subg_node = self.generate_subgraph(graph, "node", p)
        subg_edge = self.generate_subgraph(graph, "edge", p)
        element_to_graph = {}
        for k, v in subg_node.items():
            element_to_graph[k] = v

        for k, v in subg_edge.items():
            element_to_graph[k] = v
        return element_to_graph

    @staticmethod
    def solve_basic_graph(graph: nx.Graph):
        nodes = graph.nodes
        min_H = np.inf
        basic_sol = dict()
        for i in range(2 ** len(nodes)):
            cur_H = 0
            tmp_sol = dict()
            binary_list = [int(x) for x in bin(i)[2:].zfill(len(nodes))]
            for i, nd in enumerate(nodes):
                tmp_sol[nd] = binary_list[i]

            for nd in graph.nodes:
                sigma_nd = 1 if tmp_sol[nd] == 1 else -1
                cur_H += graph.nodes[nd]["weight"] * sigma_nd

            for ed in graph.edges:
                u, v = ed
                sigma_u = 1 if tmp_sol[u] == 1 else -1
                sigma_v = 1 if tmp_sol[v] == 1 else -1
                cur_H += graph.adj[u][v]["weight"] * sigma_u * sigma_v

            if cur_H < min_H:
                min_H = cur_H
                basic_sol = tmp_sol
        return basic_sol

    @staticmethod
    def get_solution(graph, basic_sol: dict = None):
        sol = defaultdict(lambda: -1)
        queue = list(basic_sol.keys()) if basic_sol is not None else []
        for i in graph.nodes:
            if sol[i] == -1:  # len(graph.neighbors(i)) != 0 and
                queue.append(i)
                while len(queue) > 0:
                    u = queue[0]
                    queue.pop(0)
                    if sol[u] == -1:
                        sol[u] = 0
                    if u not in graph:
                        continue
                    neighbor_u = list(graph.neighbors(u))
                    for v in neighbor_u:
                        if sol[v] == -1:
                            sol[v] = sol[u] if graph.adj[u][v]["weight"] > 0 else 1 - sol[u]
                            queue.append(v)
        return sol

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
        element_to_graph = self.graph_decomposition(graph=self._simple_graph, p=p)
        # element_to_graph = {(0, 1):self._simple_graph}

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

    def run_qaoa(self):
        """
        run Qcover code to solve the given problem
        Args:
            node_num: nodes number in the graphical representation of the problem
            edge_num: edges number in the graphical representation of the problem
            is_parallel: run programs in parallel or not

        Returns:
            results of the problem, which including the optimal parameters of the circuit model,
            the optimal expectation value and the number of times the optimizer iterates
        """
        # nodes_weight, edges_weight = get_graph_weights(self._simple_graph)
        # self._backend._nodes_weight = nodes_weight
        # self._backend._edges_weight = edges_weight

        x, fun, nfev = self._optimizer.optimize(objective_function=self.calculate)
        res = {"Optimal parameter value": x, "Expectation of Hamiltonian": fun, "Total iterations": nfev}
        return res

    def run_rqaoa(self, node_threshold, iter_time, corr_method):
        try:
            if iter_time < 1:
                raise UserConfigError("iter_time should be a value greater than 1")
        except UserConfigError as e:
            print(e)
            print("iter_time will be set to 1")
            iter_time = 1

        sexp = []

        node_sol = len(self._simple_graph)
        current_g = self._simple_graph
        node_num = len(current_g.nodes)
        pathg = nx.Graph()
        while node_num > node_threshold:
            # nodes_weight, edges_weight = get_graph_weights(current_g)
            # self._backend._nodes_weight = nodes_weight
            # self._backend._edges_weight = edges_weight
            self._backend._origin_graph = current_g

            optization_rounds = iter_time
            while optization_rounds > 0:
                self._optimizer.optimize(objective_function=self.calculate)  # x, fun, nfev =

                if corr_method == 'expectation':
                    # origin format of RQAOA
                    exp_sorted = sorted(self._backend.element_expectation.items(),
                                        key=lambda item: abs(item[1]),
                                        reverse=True)
                else:
                    corr = dict()
                    for key, val in self._backend.element_expectation.items():
                        if isinstance(key, tuple):
                            vi, vj = self._backend.element_expectation[key[0]], self._backend.element_expectation[key[1]]
                            corr[key] = (val - vi * vj) / (1 - vi * vj)
                        else:
                            continue
                    exp_sorted = sorted(corr.items(), key=lambda item: abs(item[1]), reverse=True)

                u, v = exp_sorted[0][0]
                exp_value = abs(exp_sorted[0][1])

                # print("iteration on %d, max expectation is %lf" % (optization_rounds, exp_value))
                # print("--------------------------------------")
                if exp_value - 1e-8 >= 0.5:
                    break
                optization_rounds -= 1

            # if exp_value < 0.5:  # if the relation between two node is small, then run search method
            #     break
            sexp.append(exp_value)
            correlation = 1 if exp_sorted[0][1] - 1e-8 > 0 else -1
            pathg.add_edge(u, v, weight=correlation)
            if u > v:
                u, v = v, u

            for nd in current_g.neighbors(v):
                if nd == u:
                    continue
                if nd not in current_g.neighbors(u):
                    current_g.add_edge(u, nd, weight=correlation * current_g.adj[v][nd]["weight"])
                else:
                    current_g.adj[u][nd]["weight"] += correlation * current_g.adj[v][nd]["weight"]
                    # current_g.adj[nd][u]["weight"] = current_g.adj[u][nd]["weight"]
            current_g.remove_node(v)
            node_num = len(current_g.nodes)

        basic_sol = None
        if len(current_g.nodes) > 1:   #node_threshold
            basic_sol = self.solve_basic_graph(current_g)

        if basic_sol is None or len(basic_sol) < node_sol:
            tmp_sol = self.get_solution(pathg, basic_sol)
            if basic_sol is None:
                solution = tmp_sol
            else:
                solution = {**tmp_sol, **basic_sol}
        else:
            solution = basic_sol
        return solution, sexp

    def run(self,
            node_num=None, edge_num=None,
            is_parallel=False,
            mode='QAQA',
            node_threshold=1,
            iter_time=1,
            corr_method='expectation'):

        if self._simple_graph is None:
            if node_num is None or edge_num is None:
                print("Error: the graph to be solved is not specified, "
                      "arguments of form (node_number, edge_number) should be given")
                return None
            self.simple_graph(node_num, edge_num)

        self._backend._is_parallel = is_parallel
        if mode == 'QAQA':
            res = self.run_qaoa()
        else:
            res = self.run_rqaoa(node_threshold, iter_time, corr_method)

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
    sol = qc.run(is_parallel=False, mode='QAQA')  #True
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




