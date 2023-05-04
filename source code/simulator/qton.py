# 
# This is a quantum computer simulator.
# This is a special variant of Qton.
# This is for powering Qcover.
# 
# Author(s): Yunheng Ma
# Timestamp: 2022-03-30
# 


import numpy as np
from random import choices

# alphabet
alp = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      ]

class Qcircuit:

    # num_qubits = 0
    # state = None
    def __init__(self, num_qubits=1, backend="statevector"):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, complex)
        self.state[0] = 1.0
        self.backend = backend

    def apply_tensor(self, gate, num_qubits, *qubits):
        # based on `numpy.tensordot` and `numpy.einsum`
        if num_qubits > len(set(qubits)):
            raise Exception('Duplicate qubits in input.')
        global alp

        a_idx = [*range(num_qubits, 2 * num_qubits)]
        b_idx = [self.num_qubits - i - 1 for i in qubits]

        rep = gate.reshape([2] * 2 * num_qubits)
        self.state = self.state.reshape([2] * self.num_qubits)
        self.state = np.tensordot(rep, self.state, axes=(a_idx, b_idx))

        s = ''.join(alp[:self.num_qubits])
        end = s
        start = ''
        for i in range(num_qubits):
            start += end[self.num_qubits - qubits[i] - 1]
            s = s.replace(start[i], '')
        start = start + s
        self.state = np.einsum(start + '->' + end, self.state).reshape(-1)
        return None

    def _apply_1q_(self, gate, qubit):
        L = int(2**qubit)
        for i in range(0, int(2**self.num_qubits), int(2**(qubit + 1))):
            for j in range(i, i + L):
                self.state[j], \
                self.state[j + L] = np.matmul(gate, [
                    self.state[j],
                    self.state[j + L]
                    ])
        return None

    def _apply_2q_(self, gate, qubit1, qubit2):
        if qubit1 == qubit2:
            raise Exception('Cannot be same qubits.')
        if qubit1 > qubit2:
            q1, q2 = qubit1, qubit2
        else:
            q1, q2 = qubit2, qubit1
        L1, L2, L3 = 2**qubit2, 2**qubit1, 2**qubit2 + 2**qubit1
        for i in range(0, 2**self.num_qubits, 2**(q1 + 1)):
            for j in range(0, 2**q1, 2**(q2 + 1)):
                for k in range(0, 2**q2):
                    step = i + j + k
                    self.state[step], \
                    self.state[step + L1], \
                    self.state[step + L2], \
                    self.state[step + L3] = np.matmul(gate, [
                        self.state[step],
                        self.state[step + L1],
                        self.state[step + L2],
                        self.state[step + L3]
                        ])
        return None

    def apply(self, gate, num_qubits, *qubits, mode):
        if mode == "tensor":
            self.apply_tensor(gate, num_qubits, qubits)
        else:
            if num_qubits == 1:
                self._apply_1q_(gate, qubits[0])
            else:
                self._apply_2q_(gate, qubits[0], qubits[1])

    def z(self, qubit):
        gate = np.array([[1, 0], [0, -1.]])
        if self.mode == "tensor":
            self.apply_tensor(gate, 1, qubit)
        else:
            self._apply_1q_(gate, qubit)
        return gate

    def h(self, qubit):
        gate = np.array(
            [[1, 1.],
             [1, -1.]]) * np.sqrt(0.5)
        if self.backend == "tensor":
            self.apply_tensor(gate, 1, qubit)
        else:
            self._apply_1q_(gate, qubit)
        return gate

    def rz(self, qubit, theta):
        gate = np.array([
            [np.exp(-1j * theta * 0.5), 0],
            [0, np.exp(1j * theta * 0.5)],
        ])
        if self.backend == "tensor":
            self.apply_tensor(gate, 1, qubit)
        else:
            self._apply_1q_(gate, qubit)
        return gate

    def rx(self, qubit, theta):
        t = theta * 0.50
        gate = np.array([
            [np.cos(t), -1j * np.sin(t)], 
            [-1j * np.sin(t), np.cos(t)]])
        if self.backend == "tensor":
            self.apply_tensor(gate, 1, qubit)
        else:
            self._apply_1q_(gate, qubit)
        return gate

    def rzz(self, qubit1, qubit2, theta):
        a, b = np.exp(-0.5j*theta), np.exp(0.5j*theta)
        gate = np.diag([a, b, b, a])
        if self.backend == "tensor":
            self.apply_tensor(gate, 2, qubit1, qubit2)
        else:
            self._apply_2q_(gate, qubit1, qubit2)
        return gate

    def sample(self, shots=1024):
        p = self.state * self.state.conj()
        N = self.state.shape[0]
        memory = choices(range(N), weights=p, k=shots)
        counts = {}
        for i in memory:
            key = format(i, '0%db' % self.num_qubits)
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
        return counts



class Qcodes:
    # codes = []
    def __init__(self, num_qubits=1):
        self.codes = ['qc = Qcircuit(%d)'%num_qubits]
        return None


    def z(self, qubit):
        self.codes.append('qc.z(%d)'%qubit)
        return None 


    def h(self, qubit):
        self.codes.append('qc.h(%d)'%qubit)
        return None


    def rx(self, theta, qubit):
        self.codes.append('qc.rx(%f, %d)'%(theta, qubit))
        return None


    def rzz(self, theta, qubit1, qubit2):
        self.codes.append('qc.rzz(%f, %d, %d)'%(theta, qubit1, qubit2))
        return None


    def run(self):
        from Qcover.simulator import Qcircuit
        dic = locals()
        for code in self.codes:
            exec(code, dic)
        return dic['qc']
