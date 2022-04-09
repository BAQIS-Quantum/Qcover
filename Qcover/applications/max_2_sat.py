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
import networkx as nx
import matplotlib.pyplot as plt
import time
from Qcover.applications.common import get_ising_matrix, get_weights_graph, random_number_list


logger = logging.getLogger(__name__)


class Max2Sat:
    """
    max-2-sat problem:
    each clause consists of two literals and a clause is satisfied if either or both literals are true.
    minimizing the number of clauses not satisfied
    """
    def __init__(self,
                 clauses: np.array = None,
                 variable_no: int = None,
                 weight_range: tuple = (1, 100),
                 seed: int = None)-> None:
        """
        Args:
            clauses (np.array): a matrix of clauses, matrix entry is 1 if the literal variable is true, 
            0 if the literal variable is false, -1 for the complement of the variable
            variable_no (int): number of variables
            (require input: clauses)
            
        Returns:
            clauses, variable_no 
        
        Example:
            clauses_matrix = [[1,1,0,0],[1,-1,0,0],[-1,1,0,0],[-1,-1,0,0],[-1,0,1,0],[-1,0,-1,0],
                          [0,1,-1,0],[0,1,0,1],[0,-1,1,0],[0,-1,-1,0],[0,0,1,1],[0,0,-1,-1]]
            variable_number = len(clauses_matrix[0])
        """

        if clauses is None:
            assert variable_no is not None
            
            self._clauses = np.array(clauses, dtype='float64')
            self._variable_no = variable_no
            self._weight_range = weight_range
            # self._seed = seed
            self._clauses = random_number_list(n=self._length,
                                                   weight_range=self._weight_range,
                                                   seed=seed)  #self._seed
        else:
            self._clauses = np.array(clauses, dtype='float64')#clauses
            self._variable_no = variable_no
            self._weight_range = weight_range
            # self._seed = seed

        self._qmatrix = None
        self._shift = None
        
    def get_Qmatrix(self):
        """
        get the Q matrix in QUBO model of number partition problem
        
        Args:
            numbers (np.array): the number set that to be divided
            
        Returns:
        q_mat (np.array): the the Q matrix of QUBO model.

        ..math::
            if a row of self._clauses contains:
            literal x and literal j are both true, f(x)=1-x[i]-x[j]+x[i]x[j]
            literal x is true, the complement of literal j, f(x)+=x[j]-x[i]x[j]
            the complement of literal i, complement of literal j, f(x)+=x[i]x[j]
        """

        set_mat = np.zeros((self._variable_no,self._variable_no))
        q_mat = np.array(set_mat, dtype='float64')
        shift = 0.0

        for a in range(len(self._clauses)): 
           indices_1 = [i for i, x in enumerate(self._clauses[a]) if x == 1.]
           if len(indices_1)>1:
               q_mat[indices_1[0]][indices_1[1]] = 0.5
               q_mat[indices_1[1]][indices_1[0]] = 0.5
               q_mat[indices_1[0]][indices_1[0]] += -1.0
               q_mat[indices_1[1]][indices_1[1]] += -1.0
               shift += 1.0
               
           indices_2 = [i for i, x in enumerate(self._clauses[a]) if x == -1.]
           if len(indices_2)>1:
               q_mat[indices_2[0]][indices_2[1]] += 0.5
               q_mat[indices_2[1]][indices_2[0]] += 0.5
        
           for i in range(self._variable_no):
              for j in range(self._variable_no):
                  if self._clauses[a][i]==1. and self._clauses[a][j]==-1.:
                      indices_3 = [i for i, x in enumerate(self._clauses[a]) if x == 1.]
                      indices_4 = [i for i, x in enumerate(self._clauses[a]) if x == -1.]
                      q_mat[indices_3[0]][indices_4[0]] += -0.5
                      q_mat[indices_4[0]][indices_3[0]] += -0.5
                      q_mat[indices_4[0]][indices_4[0]] += 1.0
           
        return q_mat, shift
        

    def max_2_sat_value(self, x, w):  
        """Compute the value of a max-2-sat problem.

        Args:
            x (numpy.ndarray): binary string as numpy array.
            number_list (numpy.ndarray): list of numbers in the instance.

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
        m2s_graph = get_weights_graph(ising_mat)
        return m2s_graph, self._shift
    