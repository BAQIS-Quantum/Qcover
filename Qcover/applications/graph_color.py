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

import time
import logging
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_regular_graph


logger = logging.getLogger(__name__)


class GraphColoring:
    """
    graph coloring problem:
    Given a graph and K colors, each vertex is required to have only one color,
    and the colors of the connected vertices are different
    """
    def __init__(self,
                 graph: nx.Graph = None,
                 node_num: int = None,
                 color_num: int = 4,
                 node_degree: int = 3,
                 weight_range: int = 1,
                 seed: int = None,
                 penalty: int = 99998
                 ):

        self._color_num = color_num
        self._P = penalty
        if graph is None:
            self._node_num = node_num
            # self._node_degree = node_degree
            # self._weight_range = weight_range
            # self._seed = seed
            self._graph = random_regular_graph(node_num=self._node_num,
                                               degree=node_degree,
                                               weight_range=weight_range,
                                               seed=seed)   # self.get_random_regular_graph()
        else:
            self._node_num = len(graph.nodes)
            self._graph = graph
            
        self._qmatrix = None
        self._shift = None

    @property
    def node_num(self):
        return self._node_num

    @property
    def color_num(self):
        return self._color_num

    @property
    def graph(self):
        return self._graph

    def update_random_graph(self, node_num, color_num, node_degree, weight_range, seed):
        self._node_num = node_num
        self._color_num = color_num

        self._graph = random_regular_graph(node_num=self._node_num,
                                           degree=node_degree,
                                           weight_range=weight_range,
                                           seed=seed)

    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of max cut problem
        Args:
            adjacent_mat (np.array):  the adjacent matrix of graph G of the problem
            cnt (int): number of vertices in the graph of the problem
            k: the number of colors that can be used
            P: an arbitrarily large number used as constant parameter

        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.
        """

        Qmat_length = self._node_num * self._color_num
        qubo_mat = -self._P * np.eye(Qmat_length)
        adj_mat = nx.adjacency_matrix(self._graph).A
        for i in range(Qmat_length):
            for j in range(Qmat_length):
                x, y = i // self._color_num, j // self._color_num
                idl, idr = x * self._color_num, (x + 1) * self._color_num - 1
                if (i >= idl and i <= idr) and (j >= idl and j <= idr):
                    if i == j:
                        qubo_mat[i][i] = -self._P
                    else:
                        qubo_mat[i][j] = self._P
                else:
                    if adj_mat[x][y] != 0:
                        for l in range(self._color_num):
                            qubo_mat[x*self._color_num + l][y*self._color_num + l] = self._P/2
               
        shift = self._P
        return qubo_mat, shift

    def run(self):
        if self._qmatrix is None:
            self._qmatrix, self._shift = self.get_Qmatrix()

        qubo_mat = self._qmatrix
        ising_mat = get_ising_matrix(qubo_mat)
        gc_graph = get_weights_graph(ising_mat)
        return gc_graph, self._shift
    