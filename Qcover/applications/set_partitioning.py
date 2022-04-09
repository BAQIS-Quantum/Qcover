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
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import time, os
from numbers import Number
import pandas as pd
from numpy.random import choice
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_number_list


logger = logging.getLogger(__name__)


class SetPartitioning:
    """
    Set Partitioning problem:
    For a finite set S, partitioning S into several subsets such that each element of S is in one and only one subset,
    find the partition that minimises the cost without violating any constraint.
    
    """
    def __init__(self,
                 element_list: list = None,
                 length: int = None,
                 weight_range: tuple = (1, 100),
                 element_set: np.array = None, #constriants from subsets
                 weight: np.array = None,
                 P: float = None,
                 seed: int = None):
        """
        Args:
            element_list (list): a list of elements 
            length (int): length of number_list
            element_set (list): a list of constraints
            weight (list): list of weight for element in number_list
            P (int): the penalty value for the penalty terms
            (require input: element_list, element_set, weight, P)
            
        Returns: 
           element_list, length, weight_range, weight, element_set, constraints, P
            
        Example:
            element_list = ['a','b','c','d','e','f']
            element_list_len = len(element_list)
            element_weight = [3,2,1,1,3,2]
            subsets = [[1,3,6],[2,3,5,6],[3,4,5],[1,2,4,6]]
            penalty = 10
        """
                
        if element_list is None and element_set is None:
            assert length is not None
            assert len(element_set) is not None
            
            self._length = length
            self._weight_range = weight_range
            self._weight = weight
            self._element_set = element_set
            self._constraints = self.constraints()
            self._P = P
            self._seed = seed
            self._element_list = random_number_list(n=self._length,
                                                   weight_range=self._weight_range,
                                                   seed=self._seed)
            
        else:
            
            for n in [element_list]:
                if isinstance(n, Number) == False:
                    self._element_list = pd.factorize(element_list)[0] 
            self._length = len(self._element_list)
            self._weight_range = (np.abs(self._element_list).min(), np.abs(self._element_list).max())
            self._seed = seed
            self._weight = weight
            self._element_set = element_set
            self._constraints = self.constraints()
            self._P = P

        self._qmatrix = None
        self._shift = None
            
    @property
    def length(self):
        return self._length

    @property
    def weight(self):
        return self._weight
    
    def constraints(self): 
        """
        If the element in the list is included in the subset, 
        the constraints matrix entry is 1, vice versa
        """
        constraints_mat = np.zeros((self._length,self._length), dtype='float64')
        for i in range(self._length): #list length
            for j in range(self._length):
                for a in range(len(self._element_set)): #number of constraints
                    if i+1 in self._element_set[a] and j+1 in self._element_set[a]: 
                        constraints_mat[a][i] = 1.0
        return constraints_mat #binary matrix

    def update_args(self, length, weight,constraints):
        self._length = length
        self._weight = weight
        self._constraints = constraints
    
    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of number partition problem

        Args:
            numbers (np.array): the number set that to be divided

        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.
        
        ..math::
        minimise x(T)Qx
        Q[i][j] = P * number of times elements i and j appear in the same constraint
        Q[i][i] = -weight[i] - P * number of ones in the ith column of the constraint matrix
        """
        
        q_mat = np.eye(self._length, dtype='float64')
        
        for i in range(self._length):
                q_mat[i][i] = self._weight[i] - self._P * np.sum(self._constraints, axis = 0)[i]                           
                for j in range(i):
                    r = 0.0
                    for a in range(len(self._element_set)): #number of constraints
                        if i+1 in self._element_set[a] and j+1 in self._element_set[a]: 
                            r += 1.0 #number of times elements i and j appear in the same constraint
                            q_mat[i][j] = self._P * r
                            q_mat[j][i] = q_mat[i][j]
        
        shift = self._P

        return q_mat, shift

    def set_partitioning_value(self, x,w): 
        """Compute the value of a packing.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            weight (numpy.ndarray): weights of elements

        Returns:
            float: value of the partitioning.
        """
        if self._qmatrix is None:
            self._qmatrix = self.get_Qmatrix()
            
        X = np.matmul(x, np.matmul(self._qmatrix, np.transpose(x)) ) 
        return X

    def run(self):
        if self._qmatrix is None:
            self._qmatrix, self._shift = self.get_Qmatrix()

        qubo_mat = self._qmatrix
        ising_mat = get_ising_matrix(qubo_mat)
        sp_graph = get_weights_graph(ising_mat)
        return sp_graph, self._shift