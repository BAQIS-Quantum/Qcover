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
import logging
import time
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_number_list #Qcover.applications.


logger = logging.getLogger(__name__)


class NumberPartition:
    """
    number partition problem:
    given the set of numbers, divided it into two parties
    so the sum of numbers in two parties are as close as possible
    """
    def __init__(self,
                 number_list: np.array = None,
                 length: int = None,
                 weight_range: tuple = (1, 100),
                 seed: int = None) -> None:

        if number_list is None:
            assert length is not None

            self._length = length
            self._weight_range = weight_range
            self._seed = seed
            self._number_list = random_number_list(n=self._length,
                                                   weight_range=self._weight_range,
                                                   seed=self._seed)
        else:
            self._number_list = number_list
            self._length = len(number_list)
            self._weight_range = (np.abs(number_list).min(), np.abs(number_list).max())
            self._seed = seed
            
        self._qmatrix = None
        self._shift = None
        
    @property
    def length(self):
        return self._length

    @property
    def weight_range(self):
        return self._weight_range

    @property
    def number_list(self):
        return self._number_list

    def update_args(self, length, weight_range):
        self._length = length
        self._weight_range = weight_range

        self._number_list = random_number_list(n=self._length, weight_range=self._weight_range, seed=self._seed)   # self.get_random_number_list()

    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of number partition problem

        Args:
            numbers (np.array): the number set that to be divided

        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.
        """
        
        all_sum = np.sum(self._number_list)
        q_mat = np.eye(self._length)

        for i in range(self._length):
            q_mat[i][i] = self._number_list[i] * (self._number_list[i] - all_sum)
            for j in range(self._length):
                if i == j:
                    continue
                q_mat[i][j] = self._number_list[i] * self._number_list[j]
                
        shift = 0.0
        
        return q_mat,shift
    
    def partition_value(self, x, number_list):  #
        """Compute the value of a partition.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            number_list (numpy.ndarray): list of numbers in the instance.

        Returns:
            float: difference squared between the two sides of the number
                partition.
        """
        diff = np.sum(number_list[x == 0]) - np.sum(number_list[x == 1])
        return diff * diff

    def run(self):
        if self._qmatrix is None:
            self._qmatrix, self._shift = self.get_Qmatrix()
            
        qubo_mat = self._qmatrix
        ising_mat = get_ising_matrix(qubo_mat)
        np_graph = get_weights_graph(ising_mat)
        return np_graph, self._shift