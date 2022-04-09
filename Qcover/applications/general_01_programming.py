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
from scipy import optimize
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_number_list

logger = logging.getLogger(__name__)


class General01Programming:
    """
    General 01 Programming:
    problems in the form (Max cx, s.t. Ax=b and/or inequality constraints, x binary)
    Any problem in linear constraints and bounded integer variables can be
    converted through a binary expansion into QUBO model
    """

    def __init__(self,
                 element_list: list = None,
                 number_list: list = None,
                 length: int = None,
                 weight_range: tuple = (1, 100),
                 weight: list = None,
                 element_set: list = None,  # coefficient of the constraints
                 signs: list = None,
                 b: np.array = None,
                 P: float = None,
                 slack_1: int = 3,
                 seed: int = None):
        """
        Args:
            element_list (list): a list of elements 
            length (int): length of element_list
            weight (list): list of weight for element in number_list
            element_set (list): a list of coefficient of constraints
            signs (list): a list of constraint equations/inequalities signs
            b (list): a list of constraint equations/inequalities values
            P (int): the penalty value for the penalty terms
            slack_1 (int): slack variable for '<=' inequalty constraints, default value is 3
            (require input: number_list, weight, element_set, signs, b, P, slack_1(optional))

         Returns:
            element_list, length, weight_range, weight, element_set, signs, constraints, b, P

        Example:
            element_list = ['a','b','c','d','e']
            element_list_len = len(element_list)
            element_weight = [6,4,8,5,5]
            coefficients = [[2,2,4,3,2],[1,2,2,1,2],[3,3,2,4,4]]
            signs = ['<=', '=', '>=']
            constants = [7,4,5]
            penalty = 10
            slack_1 = 3
        """
        self._slack_1 = slack_1
        
        for n in [element_list]:
            if isinstance(n, Number) == False:
                number_list = pd.factorize(element_list)[0] 
                
        if number_list is None and element_set is None:
            assert length is not None
            assert len(element_set) is not None
            self._length = length
            self._weight_range = weight_range
            self._weight = weight
            self._element_set = element_set
            self._signs = signs
            self._b = b
            self._P = P
            self._constraints = self.get_constraints()
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
            self._weight = weight
            self._element_set = element_set
            self._signs = signs
            self._b = b   
            self._P = P
            self._constraints = self.get_constraints()
            self._seed = seed

        self._qmatrix = None
        self._shift = None

    @property
    def length(self):
        return self._length

    @property
    def weight(self):
        return self._weight

    def get_constraints(self):
        """
        convert all constraints into equalities of the form Ax=b
        and get the A matrix from the constraints

        Args:
            element_set (list) : a list of resource requirement of project
            slack (int): slack variable coefficient, default value is 3
        Returns:
            A (np.array): the matrix of constraint coefficients

        ..math::
            if signs[l] is '=', A_list[l] equals to element_set[l]

            if signs[j] is '<=', the default slack variable slack_1
            upper bound default value is 3, if slack_1 is an even number, it should be converted
            to an odd number by adding 1. slack_1 is then binary expanded into slack_cof1,
            A_list[j] equals to element_set[l] extended by slack_cof1.

            if signs[k] is '>=', the default slack variable slack_2
            upper bound default value is b[k]+1, if slack_2 is an even number, it should be converted
            to an odd number by adding 1. slack_2 is then binary expanded into slack_cof2.
            A_list[k] equals to element_set[k]

            A_list[l], A_list[j], A_list[k] are extended by 0s,
            A_list[k] is extended by slack_cof2 such that A_list elements have the same length.
        """
        A_list = [[] for i in range(len(self._element_set))]

        index_1 = [i for i in range(len(self._signs)) if self._signs[i] == '=']
        l = int(index_1[0])
        A_list[l] = list(self._element_set[l])

        index_2 = [i for i in range(len(self._signs)) if self._signs[i] == '<=']
        j = int(index_2[0])
        y = list([int(i) for i in bin(self._slack_1)[2:]])  # binary expansion for slack variable
        y.reverse()
        if self._slack_1 % 2 == 0:
            y = list([int(i) for i in bin(self._slack_1 + 1)[2:]])
        # the coefficients of new binary variables resulted from the slack variable
        slack_cof1 = np.multiply(np.array(y), np.float_power(2, np.arange(len(y)))).tolist()
        A_list[j] = list(self._element_set[j])
        A_list[j].extend(int(i) for i in slack_cof1)

        index_3 = [i for i in range(len(self._signs)) if self._signs[i] == '>=']
        k = int(index_3[0])
        slack_2 = self._b[k] + 1  # the upper bound of the slack variable
        z = list([int(i) for i in bin(slack_2)[2:]])  # binary expansion for slack variable
        z.reverse()
        if slack_2 % 2 == 0:
            z = list([int(i) for i in bin(slack_2 + 1)[2:]])
        # the coefficients of new binary variables resulted from the slack variable
        slack_cof2 = (-1.0 * np.multiply(np.array(z), np.float_power(2, np.arange(len(z))))).tolist()
        A_list[k] = list(self._element_set[k])

        A_list[l].extend([0.0] * (len(slack_cof1) + len(slack_cof2)))
        A_list[j].extend([0.0] * (len(slack_cof2)))
        A_list[k].extend([0.0] * (len(slack_cof1)))
        A_list[k].extend(int(i) for i in slack_cof2)

        return np.array(A_list)

    def update_args(self, length, weight, A):
        self._length = length
        self._weight = weight

    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of number partition problem

        Args:
            numbers (np.array): the number set that to be divided

        Returns:
            q_mat (np.array): the the Q matrix of QUBO model.

        ..math::
            minimise x(T)Qx
            Q[i][j] = P/2 * A(T)A[i][j]
            Q[i][i] = -weight[i] + P * (A(T)A[i][i] - 2A(T)b[i])
        """
        matrix_dimension = self._constraints.shape[1]
        q_mat = np.eye(matrix_dimension, dtype='float64')

        extended_weight = list(self._weight)
        extended_weight.extend([0] * (self._constraints.shape[1] - self._length))

        A_matrix_mul = np.matmul(np.transpose(self._constraints), self._constraints)  # A(T)A
        AT_b = np.matmul(np.transpose(self._constraints), self._b)  # A(T)b

        for i in range(matrix_dimension):
            q_mat[i][i] = -extended_weight[i] + self._P * (A_matrix_mul[i][i] - 2.0 * AT_b[i])
            for j in range(matrix_dimension):
                if i == j:
                    continue
                q_mat[i][j] = self._P * A_matrix_mul[i][j]
        
        shift = self._P * np.matmul(np.transpose(self._b), self._b)

        return q_mat, shift

    def general_01_programming_value(self, x, w):
        """Compute the value of the problem.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            weight (numpy.ndarray): weights of elements

        Returns:
            float: value of the problem.
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
        gp_graph = get_weights_graph(ising_mat)
        return gp_graph, self._shift
    