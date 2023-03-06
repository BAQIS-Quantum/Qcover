import numpy as np
import networkx as nx
from typing import Optional
from collections import defaultdict

from Qcover.optimizers import Optimizer, COBYLA
from Qcover.backends import Backend, CircuitByQiskit
from Qcover.core import Qcover
import warnings
warnings.filterwarnings("ignore")


class GHZ_Generate:

    def __init__(self,
                 node_num: int,
                 p: int = 1,
                 graph: nx.Graph = None,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQiskit(),
                 ) -> None:
        self._p = p
        self._node_num = node_num

        if graph is None:
            self._original_graph = self.get_graph()
        else:
            if self._node_num != len(graph):
                print("Error: node number should be same with the one in graph to initialize")
                return
            self._original_graph = graph

        self._qc = Qcover(self._original_graph,
                          self._p,
                          optimizer=optimizer,
                          backend=backend,
                          research_obj="GHZ")

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, ap):
        self._p = ap

    @property
    def node_num(self):
        return self._node_num

    @node_num.setter
    def node_num(self, nd):
        self._node_num = nd

    @property
    def qc(self):
        return self._qc

    def get_graph(self):
        g = nx.Graph()
        g.add_node(0, weight=0)
        for i in range(1, 2 * self._node_num + 1):
            g.add_node(i, weight=0)
            g.add_edge(i - 1, i, weight=-1)
        return g

    def run(self, is_parallel=False, mode='QAQA'):
        if self._original_graph is None:
            self._original_graph = self.get_graph()

        # nx.draw(self._original_graph)
        # if self._qc is None:
        #     self._qc = Qcover(self._original_graph,
        #                       self._p,
        #                       optimizer=self._optimizer,
        #                       backend=self._backend,
        #                       research_obj="GHZ")

        sol = self._qc.run(is_parallel=is_parallel, mode=mode)  # True
        return sol


if __name__ == '__main__':
    p = 1
    node_num = 2

    opti_params = [-0.23533129, -0.51984872, -0.63338606, -0.68489186, -0.71214422,
       -0.74310004, -0.75405114, -0.75529699, -0.75482341, -0.73029963,
       -0.78895398, -0.86082996,  0.73003181,  0.71178062,  0.72944497,
        0.73066588,  0.73455175,  0.7256124 ,  0.71018419,  0.68513461,
        0.66249872,  0.55629759,  0.37117664,  0.11451713]

    from Qcover.optimizers import COBYLA, Interp, SLSQP, L_BFGS_B, GradientDescent, SPSA
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

    ghz = GHZ_Generate(node_num=node_num,
                       p=p,
                       optimizer=optc,
                       backend=qiskit_bc)  #qt, , cirq_bc, projectq_bc, qulacs_bc
    # g = ghz.get_graph()
    # ghz._g = g
    # ghz._qc = Qcover(g, p,
    #                  optimizer=optc,
    #                  backend=qiskit_bc,
    #                  research_obj="GHZ")
    sol = ghz.run()

    print("solution is:", sol)
    params = sol["Optimal parameter value"]
    # params = [2.40756694, 2.37257408, 2.34787417, 0.79040416, 0.78994118,
    #    0.73654257]
    # ghz._qc.backend._pargs = params

    res_exp = ghz._qc.backend.expectation_calculation()
    print("optimal parameter value:", ghz._qc.backend._pargs)
    print("the optimal expectation is: ", res_exp)
    out_count = ghz._qc.backend.get_result_counts(params, ghz._g)
    import matplotlib.pyplot as plt
    from qiskit.visualization import plot_histogram
    plot_histogram(out_count)
    plt.show()

