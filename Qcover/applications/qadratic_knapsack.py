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


class QadraticKnapsack:
    """
    Qadratic Knapsack problem:
    Given a set of items, each with a weight and a value, 
    determine the number of each item to include in a collection 
    so that the total weight is less than or equal to a given limit 
    and the total value is as large as possible.
    There is an interaction between pairs of projects affecting the value obtained. 
    """
    
    def __init__(self,
                 v: np.array = None,
                 length: int = None,
                 weight_range: tuple = (1, 100),
                 # weight: np.array = None,
                 element_set: list = None, #constriants from subsets
                 b: list = None,
                 P: float = None,
                 slack: int = 3,
                 seed: int = None):
        """
        Args:
            v (np.array): a matrix of values associated with choosing projects 
            length (int): number of variables
            element_set (list) : a list of resource requirement of project
            b (list): a list of the total resource budget 
            P (int): the penalty value for the penalty terms
            slack (int): slack variable, default value is 3
            (require input: v, element_set, b, P, slack(optional))
            
        Returns:
            v, length, element_set, b, P, slack, constraints
            
        Example:
            v_list = [[2,4,3,5],[4,5,1,3],[3,1,2,2],[5,3,2,4]] 
            length = len(v_list)
            subset = [8,6,5,3]
            constant = [16]
            penalty = 10
        """
        self._slack = slack
        
        if v is None and element_set is None: #and distance is None
            assert length is not None
            assert len(element_set) is not None
            
            self._length = length
            self._weight_range = weight_range
            # self._weight = weight
            self._seed = seed
            self._element_set = element_set
            self._b = b
            self._P = P
            self._constraints = self.get_constraints()
            self._v = np.array(random_number_list(n=self._length**2,
                                                   weight_range=self._weight_range,
                                                   seed=self._seed)).reshpae(self._length,self._length)

        else:
            self._v = v
            self._length = len(v)
            self._weight_range = weight_range
            # self._weight = weight
            self._seed = seed
            self._element_set = element_set
            self._b = b
            self._P = P
            self._constraints = self.get_constraints()

        self._qmatrix = None
        self._shift = None
            
    @property
    def length(self):
        return self._length
    
    def get_constraints(self): 
        """
        convert ineqality constraint into equality constraint which have the form Ax=b
        and get the A matrix from the constraint
        
        Args:
            element_set (list) : a list of resource requirement of project
            slack (int): slack variable coefficient, default value is 3
        Returns: 
            A (np.array): the matrix of constraint coefficients
        
        ..math::
            the slack variable slack default upper bound default value is 3, 
            if slack is an even number, it should be converted to an odd number by adding 1.
            slack_1 is then binary expanded into slack_cof1, 
            A_list is extended by slack_cof1.
        """        
        A_list =[]
        y = list([int(i) for i in bin(self._slack)[2:]]) #slack variable in binary
        y.reverse()
        if self._slack % 2 == 0.0:
            y = list([int(i) for i in bin(self._slack+1)[2:]])
        slack_cof1 = np.multiply(np.array(y),np.float_power(2, np.arange(len(y)))).tolist()
        A_list = list(self._element_set)
        A_list.extend(int(i) for i in slack_cof1)
        
        return np.array(A_list)
    
    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of set packing problem
        
        Args:
            numbers (np.array): the number set that to be divided
            
        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.
        
        ..math::
            minimise x(T)Qx
            Q[i][j] = -v[i][j] + P * A[i] * A[j]
            Q[i][i] = -v[i][i] + P * (A[i]**2 - 2 * b * A[i])
            
        """
        matrix_dimension = self._constraints.shape[0]
        q_mat = np.eye(matrix_dimension, dtype='float64')
        
        extended_v = [[] for i in range(matrix_dimension)]
        extended_v[0:self._length] = list(self._v)
        for i in range(self._length):
            extended_v[i].extend([0]*(matrix_dimension-self._length))
        for j in range(self._length,matrix_dimension):
            extended_v[j] = [0]*(matrix_dimension)

        for i in range(matrix_dimension):
            q_mat[i][i] = -extended_v[i][i] + self._P*((self._constraints[i])**2 - 2.0*self._b[0]*self._constraints[i]) 
            for j in range(matrix_dimension):
                if i == j:
                    continue
                q_mat[i][j] = -extended_v[i][j] + self._P*(self._constraints[i]*self._constraints[j])
        
        shift = self._P * (-2.0 * self._slack * self._b[0] + self._b[0]**2)
        return q_mat, shift

    def quadratic_knapsack_value(self, x,w): 
        """Compute the value of quadratic knapsack.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            weight (numpy.ndarray): weights of elements

        Returns:
            float: value of the knapsack.
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
        qk_graph = get_weights_graph(ising_mat)
        return qk_graph, self._shift