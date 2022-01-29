import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#test codes used in core

#General01Programming
    # element_list = ['a','b','c','d','e']
    # element_list_len = len(element_list)
    # element_weight = [6,4,8,5,5]
    # coefficients = [[2,2,4,3,2],[1,2,2,1,2],[3,3,2,4,4]]
    # signs = ['<=', '=', '>=']
    # constants = [7,4,5]
    # penalty = 10
    # slack_1 = 3
    
    # from Qcover.applications import General01Programming
    # gp = General01Programming(element_list=element_list,
    #               signs=signs,
    #               b=constants,
    #               length=element_list_len, 
    #               weight=element_weight, 
    #               element_set=coefficients,P=penalty,slack_1=slack_1)
    # ising_g, shift = gp.run()
    
#Max2Sat
    # clauses_matrix = [[1,1,0,0],[1,-1,0,0],[-1,1,0,0],[-1,-1,0,0],[-1,0,1,0],[-1,0,-1,0],
    #                   [0,1,-1,0],[0,1,0,1],[0,-1,1,0],[0,-1,-1,0],[0,0,1,1],[0,0,-1,-1]]
    # variable_number = len(clauses_matrix[0])
    
    # from Qcover.applications import Max2Sat 
    # m2s = Max2Sat(clauses=clauses_matrix,variable_no=variable_number)
    # ising_g, shift = m2s.run()
    
#MinimumVertexCover

    # adjacency_matrix = np.array([[0, 1, 1, 0, 0],
    #                           [1, 0, 0, 1, 0],
    #                           [1, 0, 0, 1, 1],
    #                           [0, 1, 1, 0, 1],
    #                           [0, 0, 1, 1, 0]])
    # penalty = 8
    # from Qcover.applications import MinimumVertexCover 
    # mvc = MinimumVertexCover(graph=g, P=penalty)
    # ising_g, shift = mvc.run()
    
#NumberPartition
    # number_list_len = 5
    
    # from Qcover.applications import NumberPartition
    # np = NumberPartition(length=number_list_len)
    # ising_g, shift = np.run()
    
#QadraticKnapsack
    # v_list = [[2,4,3,5],[4,5,1,3],[3,1,2,2],[5,3,2,4]] 
    # length = len(v_list)
    # subset = [8,6,5,3]
    # constant = [16]
    # penalty = 10
    
    # from Qcover.applications import QadraticKnapsack
    # qk = QadraticKnapsack(v=v_list, length=length, element_set=subset, b=constant, P=penalty)
    # ising_g, shift = qk.run()
    
#QadraticAssignment
    # flow_matrix = [[0,5,2],[5,0,3],[2,3,0]] 
    # distance_matrix = [[0,8,15],[8,0,13],[15,13,0]]
    # n = len(flow_matrix)
    # subsets = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9]]
    # penalty = 200
    
    # from Qcover.applications import QadraticAssignment
    # qa = QadraticAssignment(flow=flow_matrix, distance=distance_matrix, element_set=subsets, n=n, P=penalty)
    # ising_g, shift = qa.run()
    
#SetPacking
    # element_list = ['a','b','c','d']
    # element_list_len = len(element_list)
    # element_weight = [10,8,10,12]
    # # element_weight = np.ones((element_list_len,), dtype=int)
    # subsets = [[1,2],[1,3,4]] 
    # penalty = 6
    
    # from Qcover.applications import SetPacking
    # sp = SetPacking(element_list=element_list,length=element_list_len, weight=element_weight, element_set=subsets, P=penalty)
    # ising_g, shift = sp.run()
    
# SetPartitioning
    # element_list = ['a','b','c','d','e','f']
    # element_list_len = len(element_list)
    # element_weight = [3,2,1,1,3,2]
    # subsets = [[1,3,6],[2,3,5,6],[3,4,5],[1,2,4,6]]
    # penalty = 10
    
    
    # from Qcover.applications import SetPartitioning
    # sp = SetPartitioning(element_list=element_list,length=element_list_len, weight=element_weight, element_set =subsets, P=penalty)
    # ising_g, shift = sp.run()