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
import math
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_number_list


logger = logging.getLogger(__name__)


class QadraticAssignment:
    """
    Qadratic Assignment problem:
    Given n facilities and n locations along with a flow matrix (f_ij) 
    denoting the flow of material between facilities i and j.
    A distance matrix (d_ij) specifies the distance between sites i and j. 
    The optimization problem is to find an assignment of facilities to locations 
    to minimize the weighted flow across the system. 
    """
    def __init__(self,
                 flow: np.array = None,
                 distance: np.array = None,
                 n: int = None,
                 weight_range: tuple = (1, 100),
                 element_set: np.array = None, #constriants from subsets
                 P: float = None,
                 seed: int = None):
        """
        Args:
            flow (np.array): flow matrix, denoting the flow of material between facilities
            distance (np.array): distance matrix, specifies the distance between sites
            n (int): number of sites = number of facilities
            element_set (np.array): matrix of coefficient of constraints
            P (int): the penalty value for the penalty terms
            (require input: flow, distance, element_set, P)
        
        Returns:
            flow, distance, n, element_set, P
            
        Example:
            flow_matrix = [[0,5,2],[5,0,3],[2,3,0]] 
            distance_matrix = [[0,8,15],[8,0,13],[15,13,0]]
            n = len(flow_matrix)
            subsets = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9]]
            penalty = 200
        """
        
                
        if flow is None and element_set is None: #and distance is None
            assert n is not None
            assert len(element_set) is not None
            
            self._distance = distance
            self._n = n
            self._weight_range = weight_range
            self._element_set = element_set
            self._P = P
            self._seed = seed
            self._flow = np.array(random_number_list(n=self._n**2,
                                                   weight_range=self._weight_range,
                                                   seed=self._seed)).reshpae(self._n,self._n)

        else:
            self._flow = flow
            self._distance = distance
            self._n = len(flow)
            self._seed = seed
            self._element_set = element_set
            self._P = P

        self._qmatrix = None
        self._shift = None
                    
    # def penalty(self):
    #    """
    #    penalty: some percentage (75% to 150%) of the estimate of the original objective function value
       
    #    Args:
    #        x (numpy.ndarray): binary string as numpy array.
           
    #    Returns:
    #        the penalty value
    #    """
    #    # a = choice([0,1], size = [1,self.length], p=[0.5, 0.5]) 

    #    # estimation = np.dot(a, self._weight).item()*1.5
    #    # return estimation
    #    penalty = 200
    #    return penalty
    
    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of set packing problem

        Args:
            numbers (np.array): the number set that to be divided

        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.
        
        ..math::
            minimise x(T)Qx
            Q[(j-1)*n+i][(i-1)*n+j] = flow[j][i] * distance[i][j] + P
            Q[i][i] = -2 * P
        """
        matrix_dimension = self._n * self._n
        q_mat = -2.0 * self._P * np.eye(matrix_dimension, dtype='float64')
        for i in range(matrix_dimension):
           for j in range(matrix_dimension):
               x, y = i // self._n, j // self._n 
               idl, idr = x * self._n, (x + 1.0) * self._n - 1.0 
               if (i >= idl and i <= idr) and (j >= idl and j <= idr):
                   if i == j:
                       q_mat[i][i] = -2.0 * self._P
                   else:
                       q_mat[i][j] = self._P
               else:
                   for l in range(self._n):
                       for k in range(self._n):
                            q_mat[x*self._n+k][y*self._n+l] = self._flow[x][y]*self._distance[k][l] 
                            if self._flow[x][y]*self._distance[k][l] == 0.0:
                                q_mat[x*self._n+k][y*self._n+l] = self._P
        
        shift = 2.0 * self._P
               
        return q_mat, shift

    def quadratic_assignment_value(self, x,w): 
        """Compute the value of a quadratic assignment.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            number_list (numpy.ndarray): list of numbers in the instance.

        Returns:
            float: value of the assignment.
        """
        if self._qmatrix is None:
            self._qmatrix = self.get_Qmatrix()

        X = np.matmul(x, np.matmul(self._qmatrix, np.transpose(x))) 
        return X

    def run(self):
        if self._qmatrix is None:
            self._qmatrix, self._shift= self.get_Qmatrix()

        qubo_mat = self._qmatrix
        ising_mat = get_ising_matrix(qubo_mat)
        qa_graph = get_weights_graph(ising_mat)
        return qa_graph, self._shift