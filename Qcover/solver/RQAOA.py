import time
from typing import Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from Qcover.core import Qcover
from Qcover.optimizers import Optimizer, COBYLA
from Qcover.backends import Backend, CircuitByQiskit
from Qcover.utils import get_graph_weights
from Qcover.exceptions import UserConfigError
import warnings
warnings.filterwarnings("ignore")


class RQAOA:
    # pylint: disable=invalid-name
    def __init__(self,
                 graph: nx.Graph = None,
                 p: int = 1,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQiskit()
                 ) -> None:

        assert graph is not None
        self._original_graph = graph
        self._p = p
        self._qc = Qcover(self._original_graph,
                          self._p,
                          optimizer=optimizer,
                          backend=backend,
                          research_obj="QAOA")

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

    def run(self, node_threshold, iter_time=1, is_parallel=False):
        try:
            if iter_time < 1:
                raise UserConfigError("iter_time should be a value greater than 1")
        except UserConfigError as e:
            print(e)
            print("iter_time will be set to 1")
            iter_time = 1

        self._qc.backend._is_parallel = is_parallel
        node_sol = len(self._original_graph)
        current_g = self._original_graph
        node_num = len(current_g.nodes)
        pathg = nx.Graph()
        while node_num > node_threshold:
            self._qc.backend._origin_graph = current_g
            nodes_weight, edges_weight = get_graph_weights(current_g)
            self._qc.backend._nodes_weight = nodes_weight
            self._qc.backend._edges_weight = edges_weight

            optization_rounds = iter_time
            while optization_rounds > 0:
                self._qc.optimizer.optimize(objective_function=self._qc.calculate)

                exp_sorted = sorted(self._qc.backend.element_expectation.items(),
                                    key=lambda item: abs(item[1]),
                                    reverse=True)
                u, v = exp_sorted[0][0]
                exp_value = abs(exp_sorted[0][1])
                if exp_value - 1e-8 >= 0.5:
                    break
                optization_rounds -= 1

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

            current_g.remove_node(v)
            node_num = len(current_g.nodes)

        basic_sol = None
        if len(current_g.nodes) > 1:
            basic_sol = self.solve_basic_graph(current_g)

        if basic_sol is None or len(basic_sol) < node_sol:
            tmp_sol = self.get_solution(pathg, basic_sol)
            if basic_sol is None:
                solution = tmp_sol
            else:
                solution = {**tmp_sol, **basic_sol}
        else:
            solution = basic_sol
        return solution


# usage example
if __name__ == '__main__':
    p = 1
    g = nx.Graph()
    nodes = [(0, 1), (1, 1), (2, 1)]
    edges = [(0, 1, 1), (1, 2, -1)]
    # edges = [(0, 1, 3), (1, 2, 2), (0, 2, 1)]
    for nd in nodes:
        u, w = nd[0], nd[1]
        g.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
        g.add_edge(int(u), int(v), weight=int(w))

    # from Qcover.applications import MaxCut
    # mxt = MaxCut(g)
    # ising_g, shift = mxt.run()

    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    qiskit_bc = CircuitByQiskit(expectation_calc_method="statevector")

    rqc = RQAOA(g, p,
                  optimizer=optc,
                  backend=qiskit_bc)

    st = time.time()
    sol = rqc.run(node_threshold=1)
    ed = time.time()
    print("time cost is:", ed - st)
    print("solution is:", sol)