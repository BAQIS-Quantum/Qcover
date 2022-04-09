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

import os
import sys
import time
import logging
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_regular_graph


logger = logging.getLogger(__name__)


class SherringtonKirkpatrick:
    """
    Sherrington Kirkpatrick problem: The Sherrington-Kirkpatrick model is a special instance of a spin glass.
    it's defined on the complete graph with wij randomly chosen to be ±1
    """
    def __init__(self, node_num: int = 5):
        self._node_num = node_num

    @property
    def node_num(self):
        return self._node_num

    def run(self):
        ising_mat = np.zeros((self._node_num, self._node_num)) #, dtype=float
        for i in range(self._node_num):
            for j in range(self._node_num):
                if i <= j:
                    ising_mat[i][j] = np.random.choice([1, -1], 1)
                else:
                    ising_mat[i][j] = ising_mat[j][i]
        sk_graph = get_weights_graph(ising_mat)
        return sk_graph
