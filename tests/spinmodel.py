# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:30:29 2022

@author: humengjun
"""
import random

import qiskit

from qiskit import *

from qiskit import Aer

from qiskit.circuit import Parameter

from qiskit.visualization import plot_state_city 

from qiskit.visualization import plot_histogram

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize

backend = Aer.get_backend('qasm_simulator')
backend.shots = 1024*500


#set initial measured spin state
phi = np.pi/2


def create_qaoa_cir(G, theta):
    N_qubits = len(G.nodes())
    p = len(theta)//2
    qc = QuantumCircuit(N_qubits)
    
    gamma = theta[:p]
    beta = theta[p:]
    
    for i in range((N_qubits-1)//2):
        qc.h(i)
    
    qc.ry(phi, (N_qubits-1)//2)
    
    for i in range((N_qubits+1)//2, N_qubits):
        qc.h(i)
    
    for irep in range(p):
        for pair in list(G.edges()):
            qc.rzz(-2*gamma[irep], pair[0], pair[1])
            
        #for i in range(N_qubits):
         #   qc.rx(2*beta[irep], i)
        
        for i in range((N_qubits-1)//2):
            qc.rx(2*beta[irep], i)
        for i in range((N_qubits+1)//2, N_qubits):
            qc.rx(2*beta[irep], i)
        
    qc.measure_all()
    
    return qc


def Ising_obj(x, G):
    obj = 0
    for i, j in G.edges():
        if x[i] == x[j]:
            obj -= 1
        else:
            obj +=1
    return obj



def compute_expectation(counts, G):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = Ising_obj(bitstring, G)
        avg += obj * count
        sum_count += count
    return avg/sum_count


def get_expectation(G, theta, shots=1024):
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots
    
    def execute_circ(theta):
        
        qc = create_qaoa_cir(G, theta)
        counts = backend.run(qc, seed_simulator=10, nshots=1024*100).result().get_counts()
    
        return compute_expectation(counts, G)
    
    return execute_circ


def plot_counts(x):
    qc_res = create_qaoa_cir(G, x)
    counts = backend.run(qc_res, seed_simulator=10).result().get_counts()
    qc_res.draw('mpl')
    plot_histogram(counts)
    return counts


G = nx.Graph()
N=7
#nearest interaction
for i in range(N-1):
    G.add_edge(i, i+1, weight=-1)
for i in range(N):
    G.add_node(i, weight=0)

#all connection
#for i in range(N-1):
#    for j in range(i+1, N):
#        G.add_edge(i, j, weight=-1)
#for i in range(N):
    #G.add_node(i, weight=0)

# theta = [1, 1, 1, 1, 1, 1, 1, 1]   #
p = 20
theta = [random.random() for i in range(2 * p)]
exp = get_expectation(G, theta)
res = minimize(exp, theta, method="COBYLA")
print(res)

qc_res = create_qaoa_cir(G, res.x)
counts = backend.run(qc_res, seed_simulator=10).result().get_counts()
qc_res.draw('mpl')
plot_histogram(counts)
plt.show()
# P = plot_counts(res.x)


# backend1 = Aer.get_backend('statevector_simulator')
#
# job1 = backend1.run(qc)
#
# result1 = job1.result()
#
# outputstate = result1.get_statevector(qc)

#backend2 = Aer.get_backend('qasm_simulator')

#job2 = backend2.run(qc, shots=1024*500)

#result2 = job2.result()

#counts = result2.get_counts(qc)





    