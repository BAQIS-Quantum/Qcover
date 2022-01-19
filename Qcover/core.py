"""
Object to solve QAOA problems

The QAOA problems can be represented as an Ising model, and then be transformed to a DAG.
The directed acyclic graph is decomposed by a specified p value, and these subgraphs then
be transformed as circuits and be executed on simulators, using optimizer to get the
optimal parameters of the original Ising model
"""

import sys
sys.path.append(r'E:\Working_projects\QAOA\Qcover')
import time
import warnings
from typing import Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from Qcover.optimizers import Optimizer, COBYLA
from Qcover.backends import Backend, CircuitByQiskit, CircuitByTensor
from Qcover.exceptions import GraphTypeError


class Qcover:
    """
    Qcover is a QAOA solver
    """
    # pylint: disable=invalid-name

    def __init__(self,
                 graph: nx.Graph = None,
                 p: int = 1,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQiskit()
                 ) -> None:
        """initialize a instance of Qcover"""

        assert graph is not None
        self._simple_graph = graph
        self._p = p
        self._backend = backend
        self._backend._p = p
        self._optimizer = optimizer
        self._optimizer._p = p

        self._nodes_weight = []
        self._edges_weight = []
        self._hard_to_calcute = False

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
            print("Error: the argument graph should be a instance of nx.Graph or a tuple formed as (node_num, edge_num)")

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

    def get_graph_weights(self):
        """
        get the weights of nodes and edges in graph
        Args:
            self._simple_graph
        Return:
            node weights form is dict{nid1: node_weight}, edges weights form is dict{(nid1, nid2): edge_weight}
        """
        if self._nodes_weight != [] and self._edges_weight != []:
            return self._nodes_weight, self._edges_weight

        nodew = nx.get_node_attributes(self.simple_graph, 'weight')
        edw = nx.get_edge_attributes(self.simple_graph, 'weight')
        edgew = edw.copy()
        for key, val in edw.items():
            edgew[(key[1], key[0])] = val

        self._nodes_weight = nodew
        self._edges_weight = edgew
        return nodew, edgew

    def generate_subgraph(self, dtype: str, p):
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

        if self._nodes_weight == [] or self._edges_weight == []:
            self._nodes_weight, self._edges_weight = self.get_graph_weights()

        subg_dict = defaultdict(list)
        if dtype == 'node':
            for node in self.simple_graph.nodes:
                node_set = {(node, self._nodes_weight[node])}
                edge_set = set()
                for i in range(p):
                    new_nodes = { (nd2, self._nodes_weight[nd2]) for nd1 in node_set for nd2 in self.simple_graph[nd1[0]] }
                    new_edges = {(nd1[0], nd2, self._edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in self.simple_graph[nd1[0]]}
                    node_set |= new_nodes
                    edge_set |= new_edges

                subg = self.generate_weighted_graph(node_set, edge_set)
                subg_dict[node] = subg
        else:
            for edge in self.simple_graph.edges:
                node_set = {(edge[0], self._nodes_weight[edge[0]]), (edge[1], self._nodes_weight[edge[1]])}
                edge_set = {(edge[0], edge[1], self._edges_weight[edge[0], edge[1]])}

                for i in range(p):
                    new_nodes = {(nd2, self._nodes_weight[nd2]) for nd1 in node_set for nd2 in self.simple_graph[nd1[0]]}
                    new_edges = {(nd1[0], nd2, self._edges_weight[nd1[0], nd2]) for nd1 in node_set for nd2 in
                                 self.simple_graph.adj[nd1[0]]}
                    node_set |= new_nodes
                    edge_set |= new_edges

                subg = self.generate_weighted_graph(node_set, edge_set)
                subg_dict[edge] = subg
        return subg_dict

    def graph_decomposition(self, p):
        """
        according to dtype to decompose graph
        Args:
            self.simple_graph (nx.Graph): graph to be composed
            self.p (int): the p of subgraphs"""

        if p <= 0:
            warnings.warn(" the argument of p should be >= 1 in qaoa problem, "
                          "so p would be set to the default value at 1")
            p = 1

        subg_node = self.generate_subgraph("node", p)
        subg_edge = self.generate_subgraph("edge", p)
        element_to_graph = {}
        for k, v in subg_node.items():
            element_to_graph[k] = v

        for k, v in subg_edge.items():
            element_to_graph[k] = v
        return element_to_graph

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
        element_to_graph = self.graph_decomposition(p)

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
                    if not self._hard_to_calcute:
                        print(e)
                        self._hard_to_calcute = True
                    # sys.exit()

        self._backend._pargs = pargs
        self._backend._element_to_graph = element_to_graph
        return self._backend.expectation_calculation(p)

    def run(self, node_num=None, edge_num=None, is_parallel=False):
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
        if self.simple_graph is None:
            if node_num is None or edge_num is None:
                print("Error: the graph to be solved is not specified, "
                      "arguments of form (node_number, edge_number) should be given")
                return None
            self.simple_graph(node_num, edge_num)

        if self._nodes_weight == [] or self._edges_weight == []:
            self._nodes_weight, self._edges_weight = self.get_graph_weights()

        self._backend._nodes_weight = self._nodes_weight
        self._backend._edges_weight = self._edges_weight
        self._backend._is_parallel = is_parallel

        x, fun, nfev = self._optimizer.optimize(objective_function=self.calculate, p=self._p)
        res = {"Optimal parameter value:": x, "Expectation of Hamiltonian": fun, "Total iterations": nfev}
        return res


# usage example
if __name__ == '__main__':
    # node_num, edge_num = 5, 10
    # nodes, edges = Qcover.generate_graph_data(node_num, edge_num)
    # g = Qcover.generate_weighted_graph(nodes, edges)

    from Qcover.applications import MaxCut

    mxt = MaxCut(node_num=12, node_degree=3)
    ising_g = mxt.run()

    p = 5
    # ising_g = nx.Graph()
    # nodes = [(0, 3), (1, 2), (2, 1), (3, 1), (4, 3), (5, 4), (6, 5)]
    # edges = [(0, 1, 1), (0, 2, 1), (3, 1, 2), (2, 3, 3), (4, 1, 1), (4, 2, 2),
    #          (5, 4, 2), (5, 3, 1), (6, 4, 1), (6, 4, 2), (6, 3, 3), (5, 6, 2)]
    # for nd in nodes:
    #     u, w = nd[0], nd[1]
    #     ising_g.add_node(int(u), weight=int(w))
    # for ed in edges:
    #     u, v, w = ed[0], ed[1], ed[2]
    # ising_g.add_edge(int(u), int(v), weight=int(w))

    from Qcover.optimizers import GradientDescent, Interp, Fourier, COBYLA
    # the numbers in initial_point should be setted by p
    optg = GradientDescent(maxiter=50, tol=1e-7, learning_rate=0.0001)
    # optc = COBYLA(options={'tol': 1e-3, 'disp': True})  # 'maxiter': 30,
    opti = Interp(optimize_method="COBYLA", options={'tol': 1e-3, 'disp': False})
    optf = Fourier(p=p, q=4, r=2, alpha=0.6, optimize_method="COBYLA", options={'tol': 1e-3, 'disp': False})

    from Qcover.backends import CircuitByQiskit, CircuitByCirq, CircuitByQulacs, CircuitByProjectq, CircuitByTensor
    qiskit_bc = CircuitByQiskit(expectation_calc_method="statevector")   # sample
    ts_bc = CircuitByTensor()
    cirq_bc = CircuitByCirq()
    # qulacs_bc_c = CircuitByQulacs()
    pq_bc = CircuitByProjectq()

    qc_f = Qcover(ising_g, p,
                  optimizer=optf,
                  backend=CircuitByQulacs())  # ts_bc cirq_bc pq_bc qiskit_bc

    time_start = time.time()
    res_f = qc_f.run(is_parallel=False)  # True
    time_end = time.time()
    tf = time_end - time_start
    qc_f.backend.visualization()


    optc = COBYLA(options={'maxiter': res_f["Total iterations"], 'tol': 1e-3, 'disp': True})  #
    qc_c = Qcover(ising_g, p,
                      optimizer=optc,
                      backend=CircuitByQulacs())  # ts_bc cirq_bc pq_bc qiskit_bc

    time_start = time.time()
    res_c = qc_c.run(is_parallel=False)  # True
    time_end = time.time()
    qc_c.backend.visualization()


    print('COBYLA takes: ', time_end - time_start)
    print("the result of COBYLA is:\n", res_c['Expectation of Hamiltonian'])


    print('Fourier takes: ', tf)
    print("the result of Fourier is:\n", res_f['Expectation of Hamiltonian'])
