import itertools
import os
import time
from collections import defaultdict, Callable
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble, BasicAer, transpile

from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, CircuitOp, CircuitStateFn, \
    MatrixExpectation, X, Y, Z, I

from Qcover.utils import get_graph_weights
from Qcover.backends import Backend
import warnings
warnings.filterwarnings("ignore")

class CircuitByQuafu(Backend):
    """generate a instance of CircuitByQuafu"""
    def __init__(self,
                 research: str = "QAOA",
                 is_parallel: bool = None) -> None:
        """initialize a instance of CircuitByCirq"""
        super(CircuitByQuafu, self).__init__()

        self._p = None
        self._origin_graph = None
        self._is_parallel = False if is_parallel is None else is_parallel
        self._research = research

        self._nodes_weight = None
        self._edges_weight = None
        self._element_to_graph = None
        self._pargs = None
        self._expectation_path = []
        self._element_expectation = dict()

    @property
    def element_expectation(self):
        return self._element_expectation