import qiskit

from qiskit import *

from qiskit import Aer

from Qcover.core import Qcover

from typing import Optional

from Qcover.backends import Backend,CircuitByQulacs

from Qcover.optimizers import Optimizer,COBYLA

from Qcover.utils import get_graph_weights, generate_weighted_graph, generate_graph_data

import networkx as nx

import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class QAOA_OTOC:
    
    def __init__(self,
                 p: int = 1,
                 graph: nx.Graph = None,
                 optimizer: Optional[Optimizer] = COBYLA(),
                 backend: Optional[Backend] = CircuitByQulacs(),
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
        res = self._qc.run(is_parallel=is_parallel)  # True
        return res
    
    def scrambling_circuit(self, G, butterfly, theta):
        p=int(len(theta)/2)
        N_qubits = len(G.nodes())+1
        ctrbit=1
        qc = QuantumCircuit(N_qubits,ctrbit)
    
        gamma = theta[:p]
        beta = theta[p:]
    
        #Ctrbit#
        ctr=0
        qc.h(ctr)
        qc.p(-np.pi/2, ctr)
    
        #N qubits#
        for j in range(1, N_qubits):
            qc.h(j)
            
        qc.cz(ctr,1)
        qc.barrier()
    
        ### U transformation ###
        for k in range(p):
            for i in range(1, N_qubits):
                qc.rz(2 * gamma[k]*nx.get_node_attributes(G, "weight")[i],i)
        
            for pair in list(G.edges()):
                qc.rzz(2*gamma[k]*nx.get_edge_attributes(G, "weight")[(pair[0],pair[1])], pair[0], pair[1])
        
            for nd in range(1, N_qubits):
                qc.rx(2*beta[k], nd)
                
        ### perturbation ###
        qc.barrier()
        qc.x(butterfly)
        qc.barrier()

        #### U_dag transformation #####
        for k_dag in range(p):
            for nd_dag in range(1, N_qubits):
                qc.rx(-2*beta[p-1-k_dag], nd_dag)

            for pair_dag in list(G.edges()):
                qc.rzz(-2*gamma[p-1-k_dag]*nx.get_edge_attributes(G, "weight")[(pair_dag[0],pair_dag[1])], pair_dag[0], pair_dag[1])
        
            for i_dag in range(1, N_qubits):
                qc.rz(-2 * gamma[p-1-k_dag]*nx.get_node_attributes(G, "weight")[i_dag], i_dag)

        qc.barrier()
        qc.cz(ctr,1)
        
        ###measure the ctrl bit###
        qc.p(-np.pi/2, ctr)
        qc.h(ctr)
        qc.measure(0, 0)
    
        return qc
    
    def calculate_OTOC(self, G, butterfly, theta):
        
        scric=self.scrambling_circuit(G, butterfly, theta)
        backend = Aer.get_backend('qasm_simulator')
        shots = 10000
        results = execute(scric, backend=backend, shots=shots).result()
        answer = results.get_counts()
        if answer['1'] is None:
            pass
        elif answer['1']==shots:
            measure=1
        else:
            measure=(answer['1']-answer['0'])/shots
      
        return measure
    
    
if __name__ == '__main__':
    p = 10  #
    g = nx.Graph()
    ########！！！！because there is an auxiliary qubit No.0, we need to start the No. of graphs with 1.
    nodes = [(1, 0), (2, 0), (3, 0),(4, 0),(5, 0),(6, 0),(7, 0)]
    edges = [(1, 2, -1), (2, 3, -1),(3, 4, -1),(4, 5, -1),(5, 6, -1),(6, 7, -1)]

    for nd in nodes:
        u, w = nd[0], nd[1]
        g.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
        g.add_edge(int(u), int(v), weight=int(w))

    ##############  
    qcover_bc = CircuitByQulacs()
    optc = COBYLA(options={'disp': True,'tol': 1e-3, 'maxiter': 10000}) #

    qaoa_otoc = QAOA_OTOC(graph=g,
                        p=p,
                        optimizer=optc,
                        backend=qcover_bc) 

    res = qaoa_otoc.run()

    print("solution is:", res)
        
    params = res["Optimal parameter value"]

    ###########dynamics of OTOC in the circuit
    a=[]
    b=[]
    OTOC_data=[]
        
    for l in range(p):
        a.append(params[l])
        b.append(params[p+l])
        cta=a+b

        OTOC_measure = qaoa_otoc.calculate_OTOC(g,4,cta)
        OTOC_data.append(OTOC_measure)

    
    xlabel='depth'
    ylabel='OTOC value'
    plt.figure(figsize=(13,6))

    x=[]
    for i in range(p):
        x.append(i+1)

    y=OTOC_data

    plt.plot(x,y)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.show()
