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

import logging
import time
import numpy as np
import networkx as nx
import random
from numpy.random import choice
import matplotlib.pyplot as plt
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_regular_graph


logger = logging.getLogger(__name__)


class MinimumVertexCover:
    """
    MVC (Minimum Vertex Cover) problem:
    For an undirected graph, find a vertex cover (a subset of the vertices/nodes) 
    with a minimum number of vertices in the subset
    such that each edge in the graph is incident to at least one vertex in the subset.
    For the weight graph, the weight sum of vertices in the subset is minimum.
    """
    def __init__(self,
                 graph: nx.Graph = None,
                 node_num: int = None,
                 node_degree: int = 3,
                 weight_range: int = 10,
                 P: float = None,
                 seed: int = None):
        """
        Args:
            graph (nx.Graph): an networkx graph generated from input adjacency matrix 
            node_num (int): number of nodes in the graph
            node_degree (int): node degree of the graph, default value is 3
            P (int): the penalty value for the penalty terms
            (require input: adjacency matrix, P)
            
         Returns:
            node_num, graph, P
            
        Example:
             adjacency_matrix = np.array([[0, 1, 1, 0, 0],
                              [1, 0, 0, 1, 0],
                              [1, 0, 0, 1, 1],
                              [0, 1, 1, 0, 1],
                              [0, 0, 1, 1, 0]])
             g = nx.from_numpy_matrix(adjacency_matrix)
             penalty = 8
        """
        
        if graph is None:
            # generate random graph according to node_num and node_degree and weight_range
            self._node_num = node_num
            self._graph = random_regular_graph(node_num=self._node_num,
                                               degree=node_degree,
                                               weight_range=weight_range,
                                               seed=seed)
            
            self._P = P
        else:
            self._node_num = len(graph.nodes)
            self._graph = graph
            self._P = P

        self._qmatrix = None
        self._shift = None
        
    @property
    def node_num(self):
        return self._node_num

    @property
    def graph(self):
        return self._graph
    @property
    def weight_range(self):
        return self._weight_range

    def update_random_graph(self, node_num, node_degree, weight_range, seed):
        self._node_num = node_num
        self._graph = random_regular_graph(node_num=self._node_num,
                                           degree=node_degree,
                                           weight_range=weight_range,
                                        seed=seed)

    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of MVC problem
        Args:
            adjacent_mat (np.array): the adjacent matrix of graph G of the problem

        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.
        
        ..math::
            minimise x(T)Qx
            Q[i][j] = P/2 
            Q[i][i] = -weight[i] - P * node degree[i]
        """
        adj_mat = nx.adjacency_matrix(self._graph).A
        qubo_mat = np.array(adj_mat, dtype='float64')
        
        for i in range(self._node_num):
            qubo_mat[i][i] = 1 - self._P * self._graph.degree[i] #node degree
            for j in range(self._node_num):
                if i == j:
                    continue
                elif abs(adj_mat[i][j] - 0.) <= 1e-8:
                    qubo_mat[i][j] = 0.0
                else:
                    qubo_mat[i][j] = self._P / 2.0
                    
        shift = self._P
        
        return qubo_mat, shift

    def minimum_vertex_cover_value(self, x, w):
        """Compute the value of a cut.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            w (numpy.ndarray): adjacency matrix.

        Returns:
            float: value of the cut.
        """
        if self._qmatrix is None:
            self._qmatrix = self.get_Qmatrix()

        X = np.matmul(x, np.matmul(self._qmatrix, np.transpose(x))) 
        return X
    
    def run(self):
        if self._qmatrix is None:
            self._qmatrix, self._shift = self.get_Qmatrix()

        qubo_mat = self._qmatrix
        ising_mat = get_ising_matrix(qubo_mat)
        mvc_graph = get_weights_graph(ising_mat)
        return mvc_graph, self._shift