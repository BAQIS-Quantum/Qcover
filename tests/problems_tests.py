import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Qcover.applications.set_partitioning import SetPartitioning
from Qcover.applications.set_packing import SetPacking
from Qcover.applications.general_01_programming import General01Programming
from Qcover.applications.max_2_sat import Max2Sat
from Qcover.applications.minimum_vertex_cover import MinimumVertexCover
from Qcover.applications.qadratic_knapsack import QadraticKnapsack
from Qcover.applications.quadratic_assignment import QadraticAssignment
#SetPartitioning
element_list = ['a','b','c','d','e','f']
element_list_len = len(element_list)
element_weight = [3,2,1,1,3,2]
subsets = [[1,3,6],[2,3,5,6],[3,4,5],[1,2,4,6]]
penalty = 10

g = SetPartitioning(element_list=element_list,length=element_list_len, weight=element_weight, element_set =subsets, P=penalty).run()
nx.draw_networkx(g)    # should be a completed graph
plt.show()
sp = SetPartitioning(element_list=element_list,length=element_list_len, weight=element_weight, element_set =subsets, P=penalty)
print(sp.get_Qmatrix())
print(sp.constraints)

#SetPacking
# element_list = ['a','b','c','d']
# element_list_len = len(element_list)
# element_weight = [10,8,10,12]
# # element_weight = np.ones((element_list_len,), dtype=int)
# subsets = [[1,2],[1,3,4]]
# penalty = 6
#
# g = SetPacking(element_list=element_list,length=element_list_len, weight=element_weight, element_set=subsets, P=penalty).run()
# nx.draw_networkx(g)    # should be a completed graph
# plt.show()
# sp = SetPacking(element_list=element_list,length=element_list_len, weight=element_weight, element_set=subsets, P=penalty)
# print(sp.get_Qmatrix())
# print(sp.constraints())

#General01Programming
# element_list = ['a', 'b', 'c', 'd', 'e']
# element_list_len = len(element_list)
# element_weight = [6, 4, 8, 5, 5]
# coefficients = [[2, 2, 4, 3, 2], [1, 2, 2, 1, 2], [3, 3, 2, 4, 4]]
# signs = ['<=', '=', '>=']
# constants = [7, 4, 5]
# penalty = 10
# slack_1 = 3
#
# g = General01Programming(element_list=element_list,
#                          signs=signs,
#                          b=constants,
#                          length=element_list_len,
#                          weight=element_weight,
#                          element_set=coefficients, P=penalty, slack_1=slack_1).run()
# nx.draw_networkx(g)  # should be a completed graph
# plt.show()
#
# gp = General01Programming(element_list=element_list,
#                           signs=signs,
#                           b=constants,
#                           length=element_list_len,
#                           weight=element_weight,
#                           element_set=coefficients, P=penalty, slack_1=slack_1)
# print(gp.get_Qmatrix())
# print(gp._constraints)

#Max2Sat
# clauses_matrix = [[1,1,0,0],[1,-1,0,0],[-1,1,0,0],[-1,-1,0,0],[-1,0,1,0],[-1,0,-1,0],
#                       [0,1,-1,0],[0,1,0,1],[0,-1,1,0],[0,-1,-1,0],[0,0,1,1],[0,0,-1,-1]]
# variable_number = len(clauses_matrix[0])
# g = Max2Sat(clauses=clauses_matrix,variable_no=variable_number).run()
# nx.draw_networkx(g)    # should be a completed graph
# plt.show()
#
# m2s = Max2Sat(clauses=clauses_matrix,variable_no=variable_number)
# print(m2s.get_Qmatrix())

#MinimumVertexCover
# adjacency_matrix = np.array([[0, 1, 1, 0, 0],
#                               [1, 0, 0, 1, 0],
#                               [1, 0, 0, 1, 1],
#                               [0, 1, 1, 0, 1],
#                               [0, 0, 1, 1, 0]])
# g = nx.from_numpy_matrix(adjacency_matrix)
# penalty = 8
#
# mvc = MinimumVertexCover(graph=g, P=penalty)
# nx.draw_networkx(mvc.graph)
# plt.show()  # shouldn't be a completed graph
# print(mvc.get_Qmatrix())

#QadraticKnapsack
# v_list = [[2,4,3,5],[4,5,1,3],[3,1,2,2],[5,3,2,4]]
# length = len(v_list)
# subset = [8,6,5,3]
# constant = [16]
# penalty = 10
#
# g = QadraticKnapsack(v=v_list, length=length, element_set=subset, b=constant, P=penalty).run()
# nx.draw_networkx(g)    # should be a completed graph
# plt.show()
# qk = QadraticKnapsack(v=v_list, length=length, element_set=subset, b=constant, P=penalty)
# print(qk._constraints)
# print(qk.get_Qmatrix())

#QadraticAssignment
# flow_matrix = [[0,5,2],[5,0,3],[2,3,0]]
# distance_matrix = [[0,8,15],[8,0,13],[15,13,0]]
# n = len(flow_matrix)
# subsets = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9]]
# penalty = 200
#
# g = QadraticAssignment(flow=flow_matrix, distance=distance_matrix, element_set=subsets, n=n, P=penalty).run()
# nx.draw_networkx(g)    # should be a completed graph
# plt.show()
# qa = QadraticAssignment(flow=flow_matrix, distance=distance_matrix, element_set=subsets, n=n, P=penalty)
# print( qa.get_Qmatrix())


