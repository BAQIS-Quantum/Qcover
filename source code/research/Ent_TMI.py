# This file should be run by qton, which can be got at the URL below:
# https://github.com/thewateriswide/qton_2.1

from Qcover.core import Qcover

from typing import Optional

from Qcover.backends import Backend,CircuitByQulacs

from Qcover.simulator import Qcircuit

from Qcover.optimizers import Optimizer,COBYLA

from Qcover.utils import get_graph_weights, generate_weighted_graph, generate_graph_data

import networkx as nx

import numpy as np

import warnings
warnings.filterwarnings("ignore")

class QAOA_Ent_TMI:
    
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

    def QAOA_gate(self,qc, G, theta):
        
        p = len(theta)//2
        gamma = theta[:p]
        beta = theta[p:]
    
        #N qubits#
        for j in list(G.nodes()):
            qc.h(j)
    
        ### U transformation ###
        for k in range(p):
            for i in list(G.nodes()):
                qc.rz(2 * gamma[k]*nx.get_node_attributes(G, "weight")[i],i)
        
            for pair in list(G.edges()):
                qc.rzz(2*gamma[k]*nx.get_edge_attributes(G, "weight")[(pair[0],pair[1])], pair[0], pair[1])
        
            for nd in list(G.nodes()):
                qc.rx(2*beta[k], nd)
                
        return None

    def von_entropy(rho):
        e, v= np.linalg.eig(rho)
        S=-(e*np.log(e)/np.log(2)).sum().real
        return S

    def Ent_TMI_Channel(self, G, list_C, theta):
        Qubit_Num=len(list(G.nodes()))
        ##########one-qubit entanglement
        a=[]
        b=[]
        Smean01=[]
        Num_list=[]
        for n in range(Qubit_Num):
            Num_list.append(n)
        ############entanglement entropy per layer
        for i in range(len(theta)//2):  #int()
            a.append(theta[i])
            b.append(theta[int(len(theta)/2)+i])
            cta=a+b
            Sc01=[]
            for j in range(Qubit_Num):
                fin_circ1=Qcircuit(Qubit_Num,backend='density_matrix')
                self.QAOA_gate(fin_circ1, G, cta)
                fin_circ1.reduce(Num_list[j])
                rho_c1=fin_circ1.state.copy()
                S_01=self.von_entropy(rho_c1)
                Sc01.append(S_01)
        
            smean01=np.mean(Sc01)
            Smean01.append(smean01)
        

        #######qton calculate I3
        I3_list=[]
        qc=Qcircuit(Qubit_Num,backend='unitary')
        qaoag=self.QAOA_gate(qc,len(theta)/2, G, theta)
        u=qc.state.copy()
        v=qc.state.reshape(2^(Qubit_Num)*2^(Qubit_Num))/np.sqrt(2^(Qubit_Num))
        rho=np.outer(v,v.conjugate())

        ########ABCD systems
        list_A=[0]
        list_B=list(set(list(G.nodes()))-set(list_A))
        list_pair=[]
        for i in list(G.nodes()):
            list_pair.append(i+Qubit_Num)
        list_D=list(set(list_pair)-set(list_C))

        qu_ac=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_ac.state=rho
        qu_ac.reduce(list_B+list_D)
        S_ac=self.von_entropy(qu_ac.state)

        qu_ad=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_ad.state=rho
        qu_ad.reduce(list_B+list_C)
        S_ad=self.von_entropy(qu_ad.state)

        qu_cd=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_cd.state=rho
        qu_cd.reduce(list_A+list_B)
        S_cd=self.von_entropy(qu_cd.state)

        qu_a=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_a.state=rho
        qu_a.reduce(list_B+list_C+list_D)
        S_a=self.von_entropy(qu_a.state)

        qu_c=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_c.state=rho
        qu_c.reduce(list_A+list_B+list_D)
        S_c=self.von_entropy(qu_c.state)

        qu_d=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_d.state=rho
        qu_d.reduce(list_A+list_B+list_C)
        S_d=self.von_entropy(qu_d.state)

        qu_acd=Qcircuit(2*Qubit_Num,backend='density_matrix')
        qu_acd.state=rho
        qu_acd.reduce(list_B)
        S_acd=self.von_entropy(qu_acd.state)


        I3=S_a+S_c+S_d-S_cd-S_ac-S_ad+S_acd
            
        I3_list.append(I3)

        return I3_list,Smean01

if __name__ == '__main__':
    p = 5
    g = nx.Graph()
    nodes = [(0, 1), (1, 1), (2, 1),(3, 1),(4, 1),(5, 1),(6, 1)]
    edges = [(0, 1, -1), (1, 2, -1),(2, 3, -1),(3, 4, -1),(4, 5, -1),(5, 6, -1)]

    for nd in nodes:
        u, w = nd[0], nd[1]
        g.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
        g.add_edge(int(u), int(v), weight=int(w))

    qcover_bc = CircuitByQulacs()
    optc = COBYLA(options={'disp': True,'tol': 1e-3, 'maxiter': 10000})  #

    qaoa_ent_tmi = QAOA_Ent_TMI(graph=g,
                           p=p,
                           optimizer=optc,
                           backend=qcover_bc) 

    res = qaoa_ent_tmi.run()

    print("solution is:", res)
    params = res["Optimal parameter value"]
    list_C=[13,12,11]

    channel_res=qaoa_ent_tmi.Ent_TMI_Channel(g,list_C,params)

    print("entanglement entropy at every layer:",channel_res[1])
    print("tripartite information at every layer:",channel_res[0])

    