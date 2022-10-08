import networkx as nx
import itertools
import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister
import math
from collections import defaultdict
import copy
import re
import matplotlib.pyplot as plt

PI = math.pi


class CompilerforQAOA:
    """compilerforQAOA for compiling combinatorial optimization
       problem graphs to quantum hardware.
    """

    def __init__(self,
                 g: nx.Graph = None,
                 p: int = 1,
                 optimal_params: list = None,
                 api_token: str = None,
                 cloud_backend: str = "ScQ-P20") -> None:
        self._gate = "CNOT"
        self._g = g
        self._p = p
        self._optimal_params = optimal_params
        self._api_token = api_token
        self._cloud_backend = cloud_backend
        self._nodes = {(k, v['weight']) for k, v in g.nodes()._nodes.items()}
        self._edges = {(u, v, g.adj[u][v]['weight']) for u, v in list(g.edges())}
        self._logical_qubits = len(g.nodes())
        self._physical_qubits = self._logical_qubits

    def random_layout_mapping(self, physical_qubits, logical_qubits):
        """
        Args:
            physical_qubits (int): The number of hardware qubits
            logical_qubits (int): The number of qubits in a circuit
        Returns:
            qubits_mapping (dict): Random mapping of logical qubits to physical qubits
        """
        if physical_qubits >= logical_qubits:
            lq = QuantumRegister(logical_qubits, 'p')
            random_physical_qubits = random.sample(range(0, physical_qubits), logical_qubits)
            qubits_mapping = {}
            for i in range(0, logical_qubits):
                qubits_mapping[random_physical_qubits[i]] = lq[i]
                x = sorted(list(qubits_mapping.keys()))
                qubits_mapping = {x[i]: qubits_mapping[x[i]] for i in range(len(x))}
            return qubits_mapping
        else:
            print("Error: physical qubits must be larger than logical qubits")

    def simple_layout_mapping(self, phys_qubits_order=None):
        """
        Args:
            physical_qubits (int): The number of hardware qubits
            logical_qubits (int): The number of qubits in a circuit
            phys_qubits_order (list): The order of the selected physical qubits
        Returns:
            qubits_mapping (dict): Simple mapping of logical qubits to physical qubits
        """
        if self._physical_qubits >= self._logical_qubits:
            lq = QuantumRegister(self._logical_qubits, 'p')
            phys_qubits_order = [i for i in
                                 range(0, self._logical_qubits)] if phys_qubits_order is None else phys_qubits_order
            qubits_mapping = {}
            for i in range(0, self._logical_qubits):
                qubits_mapping[phys_qubits_order[i]] = lq[i]
                x = sorted(list(qubits_mapping.keys()))
                qubits_mapping = {x[i]: qubits_mapping[x[i]] for i in range(len(x))}
            return qubits_mapping
        else:
            print("Error: physical qubits must be larger than logical qubits")

    def QAOA_logical_circuit(self):
        # p = self._p
        graph = self._g
        params = self._opti_params
        # get nodes and edges from graph
        nodes = {(k,v['weight']) for k,v in graph.nodes()._nodes.items()}
        edges = {(u,v,graph.adj[u][v]['weight']) for u,v in list(graph.edges())}
        lc = QuantumRegister(len(nodes), 'l')
        logical_circ = QuantumCircuit(lc)
        graph_edges = dict([(tuple(sorted(list(edges)[i][0:2])), list(edges)[i][2]) for i in range(0, len(edges))])
        graph_nodes = dict(nodes)
        gamma, beta = params[:self._p], params[self._p:]
        for i in range(len(graph_nodes)):
            logical_circ.h(i)
        for k in range(0, self._p):
            logical_circ.barrier()
            for u, v in graph_nodes.items():
                logical_circ.rz(2 * gamma[k] * v, u)
            for u, v in graph_edges.items():
                logical_circ.rzz(2 * gamma[k] * v, u[0], u[1])
            for u, v in graph_nodes.items():
                logical_circ.rx(2 * beta[k], u)
        return logical_circ

    def sorted_nodes_degree(self):
        """
        Args:
            graph (nx.Graph): find the node with the highest node degree in the graph
        Returns:
            sort_nodes (np.array): nodes are sorted by node degree in descending order
        """
        node_degree = dict(self._g.degree)
        sort_degree = sorted(node_degree.items(), key=lambda kv: kv[1], reverse=True)
        # sort_nodes = np.array([sort_degree[i][0] for i in range(len(sort_degree))])
        sort_nodes = [sort_degree[i][0] for i in range(len(sort_degree))]
        return sort_nodes

    def scheduled_pattern_rzz(self, qubits_mapping):
        """
        Get the fixed execution pattern of the QAOA circuit.
        Args:
            logical_qubits (int): The number of qubits in a circuit
            qubits_mapping (dict): {physical qubit: logical qubit}
                        example: {0: Qubit(QuantumRegister(6, 'p'), 1),
                                 1: Qubit(QuantumRegister(6, 'p'), 2),
                                 2: Qubit(QuantumRegister(6, 'p'), 0)}
        Returns:
            pattern_rzz_swap (dict): {k: (q1,q2), ...}, k-th execution cycle,
                                     execute rzz/swap gate between logic bit q1 and q2.
                                     gates execution pattern: rzz,rzz,swap,swap,rzz,rzz,swap,swap, ...
            rzz_gates_cycle (dict): rzz gate execution in k-th cycle, {(q1,q2): k, ...}
        """
        loop = 1
        cycle = 0
        pattern_rzz_swap = defaultdict(list)
        rzz_gates_cycle = defaultdict(list)
        mapping = qubits_mapping.copy()
        m = sorted(list(mapping.keys()))
        while loop <= math.ceil(self._logical_qubits / 2):
            r1 = 0
            while r1 < self._logical_qubits - 1:
                pattern_rzz_swap[cycle].append((mapping[m[r1]]._index, mapping[m[r1 + 1]]._index))
                rzz_gates_cycle[(mapping[m[r1]]._index, mapping[m[r1 + 1]]._index)] = cycle
                r1 = r1 + 2
            cycle = cycle + 1
            r2 = 1
            while r2 < self._logical_qubits - 1:
                pattern_rzz_swap[cycle].append((mapping[m[r2]]._index, mapping[m[r2 + 1]]._index))
                rzz_gates_cycle[(mapping[m[r2]]._index, mapping[m[r2 + 1]]._index)] = cycle
                r2 = r2 + 2
            cycle = cycle + 1
            if loop == math.ceil(self._logical_qubits / 2):
                break
            else:
                s1 = 1
                while s1 < self._logical_qubits - 1:
                    pattern_rzz_swap[cycle].append((mapping[m[s1]]._index, mapping[m[s1 + 1]]._index))
                    x = mapping[m[s1]]
                    mapping[m[s1]] = mapping[m[s1 + 1]]
                    mapping[m[s1 + 1]] = x
                    s1 = s1 + 2
                cycle = cycle + 1
                s2 = 0
                while s2 < self._logical_qubits - 1:
                    pattern_rzz_swap[cycle].append((mapping[m[s2]]._index, mapping[m[s2 + 1]]._index))
                    x = mapping[m[s2]]
                    mapping[m[s2]] = mapping[m[s2 + 1]]
                    mapping[m[s2 + 1]] = x
                    s2 = s2 + 2
                cycle = cycle + 1
            loop = loop + 1
        if self._logical_qubits % 2 == 1:
            pattern_rzz_swap.pop(cycle - 1)
        return pattern_rzz_swap, rzz_gates_cycle

    def QAOA_physical_circuit(self, pattern_rzz_swap, qubits_mapping):
        """
        Get the fixed execution pattern of the QAOA circuit.
        Args:
            nodes (set): The set of nodes of the weight graph, the elements are (node, weight)
            edges (set): The set of edges of the weight graph, the elements are (node1, node2, weight)
            params (numpy.array): gamma and beta parameters of QAOA circuits
            pattern_rzz_swap (dict): Execution pattern of rzz and swap gates
            qubits_mapping (dict): {physical bit (int): logical bit (Qubit(QuantumRegister(6, 'p'), 1)), ...}
            p (int): Layers of QAOA circuits
        Returns:
            circuit: QAOA physical circuit #qiskit
            final_gates_scheduled (dict):
        """
        # p = self._p
        graph_edges = dict([(tuple(sorted(list(self._edges)[i][0:2])), list(self._edges)[i][2]) for i in range(0, len(self._edges))])
        graph_nodes = dict(self._nodes)
        gamma, beta = self._opti_params[:self._p], self._opti_params[self._p:]
        mapping = qubits_mapping.copy()
        qubits_mapping_initial = qubits_mapping.copy()
        gates_scheduled = defaultdict(list)
        m = sorted(list(mapping.keys()))

        depth = 0
        for i in range(len(graph_nodes)):
            u = mapping[m[i]]._index
            gates_scheduled[depth].append(('Rz', (u, graph_nodes[u])))
        depth = depth + 1

        loop = 1
        cycle = 0
        while loop <= math.ceil(len(graph_nodes) / 2):
            r1 = 0
            for i in range(len(pattern_rzz_swap[cycle])):
                if tuple(sorted(pattern_rzz_swap[cycle][i])) in list(graph_edges.keys()):
                    edges_weight = graph_edges[tuple(sorted(pattern_rzz_swap[cycle][i]))]
                    gates_scheduled[depth + cycle].append(('Rzz', (pattern_rzz_swap[cycle][i], edges_weight)))
                    graph_edges.pop(tuple(sorted(pattern_rzz_swap[cycle][i])))
                r1 = r1 + 2
            cycle = cycle + 1
            if cycle == len(pattern_rzz_swap) or not graph_edges:
                break
            r2 = 1
            for i in range(len(pattern_rzz_swap[cycle])):
                if tuple(sorted(pattern_rzz_swap[cycle][i])) in list(graph_edges.keys()):
                    edges_weight = graph_edges[tuple(sorted(pattern_rzz_swap[cycle][i]))]
                    gates_scheduled[depth + cycle].append(('Rzz', (pattern_rzz_swap[cycle][i], edges_weight)))
                    graph_edges.pop(tuple(sorted(pattern_rzz_swap[cycle][i])))
                r2 = r2 + 2
            cycle = cycle + 1
            if cycle == len(pattern_rzz_swap) or not graph_edges:
                break
            s1 = 1
            for i in range(len(pattern_rzz_swap[cycle])):
                gates_scheduled[depth + cycle].append(('SWAP', pattern_rzz_swap[cycle][i]))
                x = mapping[m[s1]]
                mapping[m[s1]] = mapping[m[s1 + 1]]
                mapping[m[s1 + 1]] = x
                s1 = s1 + 2
            cycle = cycle + 1
            s2 = 0
            for i in range(len(pattern_rzz_swap[cycle])):
                gates_scheduled[depth + cycle].append(('SWAP', pattern_rzz_swap[cycle][i]))
                x = mapping[m[s2]]
                mapping[m[s2]] = mapping[m[s2 + 1]]
                mapping[m[s2 + 1]] = x
                s2 = s2 + 2
            cycle = cycle + 1
            loop = loop + 1

        depth = depth + cycle
        for i in range(len(graph_nodes)):
            gates_scheduled[depth].append(('Rx', (mapping[m[i]]._index, 1)))

        first_gates_scheduled = defaultdict(list)
        layer_keys = sorted(list(gates_scheduled.keys()))
        for i in range(len(layer_keys)):
            first_gates_scheduled[i] = gates_scheduled[layer_keys[i]]

        # first_gates_scheduled: The first layer of circuit
        # final_gates_scheduled: Construct the entire QAOA circuit through the first layer of circuit
        final_gates_scheduled = defaultdict(list)
        a = len(first_gates_scheduled)
        lq = QuantumRegister(len(graph_nodes), qubits_mapping[list(qubits_mapping.keys())[0]]._register.name)
        circuit = QuantumCircuit(lq)
        depth = 0
        for i in range(len(graph_nodes)):
            mh = list(qubits_mapping_initial.keys())
            circuit.h(qubits_mapping_initial[mh[i]]._index)
            final_gates_scheduled[depth].append(('H', mh[i], qubits_mapping_initial[mh[i]]._index))
        for k in range(0, self._p):
            circuit.barrier()
            if k % 2 == 0:
                direction = list(range(0, a))
            else:
                direction = list(range(a - 2, 0, -1))
                direction.insert(0, 0)
                direction.append(a - 1)
            for i in direction:
                depth = depth + 1
                for j in range(len(first_gates_scheduled[i])):
                    if first_gates_scheduled[i][j][0] == 'Rz':
                        u = first_gates_scheduled[i][j][1][0]
                        nodes_weight = first_gates_scheduled[i][j][1][1]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[lq[u]]
                        circuit.rz(2 * gamma[k] * nodes_weight, m.index(u))
                        final_gates_scheduled[depth].append(
                            ('Rz', (u, 2 * gamma[k] * nodes_weight),
                             (qubits_mapping_initial[u]._index, 2 * gamma[k] * nodes_weight)))

                    if first_gates_scheduled[i][j][0] == 'Rzz':
                        u, v = first_gates_scheduled[i][j][1][0]
                        edges_weight = first_gates_scheduled[i][j][1][1]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[lq[u]]
                        v = {v: k for k, v in qubits_mapping_initial.items()}[lq[v]]
                        circuit.rzz(2 * gamma[k] * edges_weight, m.index(u), m.index(v))
                        final_gates_scheduled[depth].append(
                            ('Rzz', ((u, v), 2 * gamma[k] * edges_weight),
                             ((qubits_mapping_initial[u]._index, qubits_mapping_initial[v]._index),
                              2 * gamma[k] * edges_weight)))

                    if first_gates_scheduled[i][j][0] == 'SWAP':
                        u, v = first_gates_scheduled[i][j][1]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[lq[u]]
                        v = {v: k for k, v in qubits_mapping_initial.items()}[lq[v]]
                        circuit.swap(m.index(u), m.index(v))
                        final_gates_scheduled[depth].append(
                            ('SWAP', (u, v),
                             (qubits_mapping_initial[u]._index, qubits_mapping_initial[v]._index)))
                        x = qubits_mapping_initial[v]
                        qubits_mapping_initial[v] = qubits_mapping_initial[u]
                        qubits_mapping_initial[u] = x

                    if first_gates_scheduled[i][j][0] == 'Rx':
                        u = first_gates_scheduled[i][j][1][0]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[lq[u]]
                        circuit.rx(2 * beta[k], m.index(u))
                        final_gates_scheduled[depth].append(
                            ('Rx', (u, 2 * beta[k]), (qubits_mapping_initial[u]._index, 2 * beta[k])))

        # Get the final gates scheduling sequence
        rearrange_gates_scheduled = copy.deepcopy(final_gates_scheduled)
        for i in range(len(final_gates_scheduled) - 1):
            list_bits = [i for i in range(len(final_gates_scheduled[0]))]
            if final_gates_scheduled[i]:
                if final_gates_scheduled[i][0][0] == 'SWAP':
                    list_bits = [final_gates_scheduled[i][k][1] for k in range(len(final_gates_scheduled[i]))]
                    list_bits = [n for item in list_bits for n in item]
                elif final_gates_scheduled[i][0][0] == 'Rzz':
                    list_bits = [final_gates_scheduled[i][k][1][0] for k in range(len(final_gates_scheduled[i]))]
                    list_bits = [n for item in list_bits for n in item]
                else:
                    pass
            for k in range(i + 1, len(final_gates_scheduled)):
                if final_gates_scheduled[k]:
                    if final_gates_scheduled[k][0][0] == 'SWAP':
                        for j in range(len(final_gates_scheduled[k])):
                            bits = final_gates_scheduled[k][j][1]
                            if (bits[0] not in list_bits) and (bits[1] not in list_bits):
                                rearrange_gates_scheduled[i].append(final_gates_scheduled[k][j])
                                rearrange_gates_scheduled[k].remove(final_gates_scheduled[k][j])
                                list_bits.extend(bits)
                                list_bits = list(set(list_bits))
                            elif (bits[0] in list_bits) or (bits[1] in list_bits):
                                list_bits.extend(bits)
                                list_bits = list(set(list_bits))
                            else:
                                pass
                    elif final_gates_scheduled[k][0][0] == 'Rzz':
                        for j in range(len(final_gates_scheduled[k])):
                            bits = final_gates_scheduled[k][j][1][0]
                            if (bits[0] not in list_bits) and (bits[1] not in list_bits):
                                rearrange_gates_scheduled[i].append(final_gates_scheduled[k][j])
                                rearrange_gates_scheduled[k].remove(final_gates_scheduled[k][j])
                                list_bits.extend(bits)
                                list_bits = list(set(list_bits))
                            elif (bits[0] in list_bits) or (bits[1] in list_bits):
                                list_bits.extend(bits)
                                list_bits = list(set(list_bits))
                            else:
                                pass
                    else:
                        pass
                    final_gates_scheduled = copy.deepcopy(rearrange_gates_scheduled)
                if len(list_bits) >= len(final_gates_scheduled[0]) - 1:
                    break
        k = 0
        final_gates_scheduled = defaultdict(list)
        for i in range(len(rearrange_gates_scheduled)):
            if rearrange_gates_scheduled[i]:
                final_gates_scheduled[k] = rearrange_gates_scheduled[i]
                k = k + 1

        return circuit, final_gates_scheduled, qubits_mapping_initial


    def gates_decomposition(self, final_gates_scheduled):
        # gates decomposition
        hq = QuantumRegister(self._physical_qubits, 'q')
        hardware_circuit = QuantumCircuit(hq)
        hardware_gates_scheduled = list([])
        for i in range(len(final_gates_scheduled)):
            layer_list = list([])
            layer_list1 = list([])
            layer_list2 = list([])
            for j in range(len(final_gates_scheduled[i])):
                if final_gates_scheduled[i][j][0] == 'H':
                    u = final_gates_scheduled[i][j][1]
                    layer_list.append(['H', u, 0])
                    hardware_circuit.h(u)
                if final_gates_scheduled[i][j][0] == 'Rz':
                    u, theta = final_gates_scheduled[i][j][1]
                    layer_list.append(['Rz', u, theta])
                    hardware_circuit.rz(theta, u)
                if final_gates_scheduled[i][j][0] == 'Rx':
                    u, theta = final_gates_scheduled[i][j][1]
                    layer_list.append(['Rx', u, theta])
                    hardware_circuit.rx(theta, u)
                if final_gates_scheduled[i][j][0] == 'Rzz':
                    for k in range(3):
                        if k == 0:
                            u, v = final_gates_scheduled[i][j][1][0]
                            layer_list.append(['CNOT', [u, v]])
                            hardware_circuit.cx(u, v)
                        if k == 1:
                            u, v = final_gates_scheduled[i][j][1][0]
                            theta = final_gates_scheduled[i][j][1][1]
                            layer_list1.append(['Rz', v, theta])
                            hardware_circuit.rz(theta, v)
                        if k == 2:
                            u, v = final_gates_scheduled[i][j][1][0]
                            layer_list2.append(['CNOT', [u, v]])
                            hardware_circuit.cx(u, v)
                if final_gates_scheduled[i][j][0] == 'SWAP':
                    for k in range(3):
                        if k == 0:
                            u, v = final_gates_scheduled[i][j][1]
                            ma, mi = max(u, v), min(u, v)
                            layer_list.append(['CNOT', [mi, ma]])
                            hardware_circuit.cx(mi, ma)
                        if k == 1:
                            u, v = final_gates_scheduled[i][j][1]
                            ma, mi = max(u, v), min(u, v)
                            layer_list1.append(['CNOT', [ma, mi]])
                            hardware_circuit.cx(ma, mi)
                        if k == 2:
                            u, v = final_gates_scheduled[i][j][1]
                            ma, mi = max(u, v), min(u, v)
                            layer_list2.append(['CNOT', [mi, ma]])
                            hardware_circuit.cx(mi, ma)
            if layer_list:
                hardware_gates_scheduled.append(layer_list)
            if layer_list1:
                hardware_gates_scheduled.append(layer_list1)
            if layer_list2:
                hardware_gates_scheduled.append(layer_list2)
        return hardware_gates_scheduled, hardware_circuit

    def cnot_gates_optimization(self, hardware_gates_scheduled, physical_qubits=None):
        # CNOT gates optimization
        # Two identical CNOT gates adjacent to each other are eliminated:
        # CNOT(i,j)CNOT(i,j) = identity matrix
        depth = len(hardware_gates_scheduled)
        opt_hardware_gates_scheduled = copy.deepcopy(hardware_gates_scheduled)
        for i in range(depth - 1):
            next_list = [hardware_gates_scheduled[i + 1][k][1] for k in
                         range(len(hardware_gates_scheduled[i + 1]))]
            for j in range(len(hardware_gates_scheduled[i])):
                if hardware_gates_scheduled[i][0][0] == 'CNOT':
                    if hardware_gates_scheduled[i][j][1] in next_list:
                        opt_hardware_gates_scheduled[i].remove(hardware_gates_scheduled[i][j])
                        opt_hardware_gates_scheduled[i + 1].remove(hardware_gates_scheduled[i][j])
        opt_hardware_gates_scheduled = list(filter(None, opt_hardware_gates_scheduled))

        if physical_qubits is not None:
            oq = QuantumRegister(physical_qubits, 'q')
            optimized_circuit = QuantumCircuit(oq)
            for i in range(len(opt_hardware_gates_scheduled)):
                for j in range(len(opt_hardware_gates_scheduled[i])):
                    if opt_hardware_gates_scheduled[i][j][0] == 'H':
                        optimized_circuit.h(opt_hardware_gates_scheduled[i][j][1])
                    if opt_hardware_gates_scheduled[i][j][0] == 'Rz':
                        _, v, theta = opt_hardware_gates_scheduled[i][j]
                        optimized_circuit.rz(theta, v)
                    if opt_hardware_gates_scheduled[i][j][0] == 'Rx':
                        _, v, theta = opt_hardware_gates_scheduled[i][j]
                        optimized_circuit.rx(theta, v)
                    if opt_hardware_gates_scheduled[i][j][0] == 'CNOT':
                        _, q = opt_hardware_gates_scheduled[i][j]
                        optimized_circuit.cx(q[0], q[1])
        else:
            optimized_circuit = None
        return opt_hardware_gates_scheduled, optimized_circuit

    def iswap_gates_scheduled_new(self, final_gates_scheduled, b=4):
        # gates decomposition
        hq = QuantumRegister(self._physical_qubits, 'q')
        hardware_circuit = QuantumCircuit(hq)
        iswap_gates_list = list([])
        # hardware_gates_list = defaultdict(list)
        # gate_index = 0
        for i in range(len(final_gates_scheduled)):
            gate_name = [gate[0] for gate in final_gates_scheduled[i]]
            count = {}
            for name in gate_name:
                count[name] = count.get(name, 0) + 1
            if 'Rzz' in gate_name and 'SWAP' not in gate_name:
                layer_list = defaultdict(list)
                for j in range(len(final_gates_scheduled[i])):
                    u, v = final_gates_scheduled[i][j][1][0]
                    theta = final_gates_scheduled[i][j][1][1]
                    layer_list = self.rzz_to_iswap(layer_list, u, v, theta, b=b)
                for ll in range(len(layer_list)):
                    iswap_gates_list.extend(layer_list[ll])

            elif 'Rzz' not in gate_name and 'SWAP' in gate_name:
                layer_list = defaultdict(list)
                for j in range(len(final_gates_scheduled[i])):
                    u, v = final_gates_scheduled[i][j][1]
                    layer_list = self.swap_to_iswap(layer_list, u, v, b=4)
                for ll in range(len(layer_list)):
                    iswap_gates_list.extend(layer_list[ll])

            elif 'Rzz' in gate_name and 'SWAP' in gate_name:
                rzz_node = [m for m, x in enumerate(gate_name) if x == 'Rzz']
                swap_node = [m for m, x in enumerate(gate_name) if x == 'SWAP']
                if count['Rzz'] > count['SWAP']:
                    # Rzz gate
                    layer_list = defaultdict(list)
                    for j in rzz_node:
                        u, v = final_gates_scheduled[i][j][1][0]
                        theta = final_gates_scheduled[i][j][1][1]
                        layer_list = self.rzz_to_iswap(layer_list, u, v, theta, b=b)
                    for ll in range(len(layer_list)):
                        iswap_gates_list.extend(layer_list[ll])
                    # SWAP gate
                    layer_list = defaultdict(list)
                    for j in swap_node:
                        u, v = final_gates_scheduled[i][j][1]
                        layer_list = self.swap_to_iswap(layer_list, u, v, b=4)
                    for ll in range(len(layer_list)):
                        iswap_gates_list.extend(layer_list[ll])
                else:
                    # SWAP gate
                    layer_list = defaultdict(list)
                    for j in swap_node:
                        u, v = final_gates_scheduled[i][j][1]
                        layer_list = self.swap_to_iswap(layer_list, u, v, b=4)
                    for ll in range(len(layer_list)):
                        iswap_gates_list.extend(layer_list[ll])
                    # Rzz gate
                    layer_list = defaultdict(list)
                    for j in rzz_node:
                        u, v = final_gates_scheduled[i][j][1][0]
                        theta = final_gates_scheduled[i][j][1][1]
                        layer_list = self.rzz_to_iswap(layer_list, u, v, theta, b=b)
                    for ll in range(len(layer_list)):
                        iswap_gates_list.extend(layer_list[ll])

            else:
                for j in range(len(final_gates_scheduled[i])):
                    if final_gates_scheduled[i][j][0] == 'H':
                        u = final_gates_scheduled[i][j][1]
                        iswap_gates_list.append(['H', u, 0])
                        hardware_circuit.h(u)
                    elif final_gates_scheduled[i][j][0] == 'Rz':
                        u, theta = final_gates_scheduled[i][j][1]
                        iswap_gates_list.append(['Rz', u, round(theta, b)])
                        hardware_circuit.rz(theta, u)
                    elif final_gates_scheduled[i][j][0] == 'Rx':
                        u, theta = final_gates_scheduled[i][j][1]
                        iswap_gates_list.append(['Rx', u, round(theta, b)])
                        hardware_circuit.rx(theta, u)
                    else:
                        pass
        return iswap_gates_list, hardware_circuit

    def rzz_to_iswap(self, layer_list, u, v, theta, b=4):
        # layer_list = defaultdict(list)
        layer_list[0].append(['Rz', u, round(-PI / 2, b)])
        layer_list[0].append(['Rx', v, round(PI / 2, b)])
        layer_list[1].append(['Rz', v, round(PI / 2, b)])
        layer_list[2].append(['iSWAP', [u, v]])
        layer_list[3].append(['Rx', u, round(PI / 2, b)])
        layer_list[4].append(['iSWAP', [u, v]])
        layer_list[5].append(['Rz', v, round(PI / 2, b)])
        layer_list[6].append(['Rz', v, round(theta, b)])
        layer_list[7].append(['Rz', u, round(-PI / 2, b)])
        layer_list[7].append(['Rx', v, round(PI / 2, b)])
        layer_list[8].append(['Rz', v, round(PI / 2, b)])
        layer_list[9].append(['iSWAP', [u, v]])
        layer_list[10].append(['Rx', u, round(PI / 2, b)])
        layer_list[11].append(['iSWAP', [u, v]])
        layer_list[12].append(['Rz', v, round(PI / 2, b)])
        return layer_list

    def swap_to_iswap(self, layer_list, u, v, b=4):
        # layer_list = defaultdict(list)
        ma, mi = max(u, v), min(u, v)
        layer_list[0].append(['iSWAP', [mi, ma]])
        layer_list[1].append(['Rx', ma, round(-PI / 2, b)])
        layer_list[2].append(['iSWAP', [mi, ma]])
        layer_list[3].append(['Rx', mi, round(-PI / 2, b)])
        layer_list[4].append(['iSWAP', [mi, ma]])
        layer_list[5].append(['Rx', ma, round(-PI / 2, b)])
        return layer_list

    def iswap_finall_scheduled(self, iswap_gates_list):
        # get iswap_gates_scheduled
        gates_list = defaultdict(list)
        for i in range(len(iswap_gates_list)):
            gates_list[i] = [iswap_gates_list[i]]
        rearrange_gates_scheduled = copy.deepcopy(gates_list)

        for i in range(len(gates_list) - 1):
            # list_bits = [i for i in range(logical_qubits)]
            list_bits = []
            if gates_list[i]:
                if isinstance(gates_list[i][0][1], int):
                    list_bits.extend([gates_list[i][0][1]])
                if isinstance(gates_list[i][0][1], list):
                    list_bits.extend(gates_list[i][0][1])
            count = -1
            for k in range(i + 1, len(gates_list)):
                count += 1
                if gates_list[k]:
                    if isinstance(gates_list[k][0][1], int) and (gates_list[k][0][1] not in list_bits):
                        rearrange_gates_scheduled[i].append(gates_list[k][0])
                        rearrange_gates_scheduled[k].remove(gates_list[k][0])
                        list_bits.extend([gates_list[k][0][1]])
                        list_bits = list(set(list_bits))
                    if isinstance(gates_list[k][0][1], list):
                        if (gates_list[k][0][1][0] not in list_bits) and (gates_list[k][0][1][1] not in list_bits):
                            rearrange_gates_scheduled[i].append(gates_list[k][0])
                            rearrange_gates_scheduled[k].remove(gates_list[k][0])
                            list_bits.extend(gates_list[k][0][1])
                            list_bits = list(set(list_bits))
                        elif (gates_list[k][0][1][0] in list_bits) or (gates_list[k][0][1][1] in list_bits):
                            list_bits.extend(gates_list[k][0][1])
                            list_bits = list(set(list_bits))
                        else:
                            pass
                    gates_list = copy.deepcopy(rearrange_gates_scheduled)
                if (len(list_bits) == self._logical_qubits):
                    # if (len(list_bits) == logical_qubits) or count > 2*(logical_qubits * 27):
                    break
        k = 0
        iswap_gates_scheduled = defaultdict(list)
        for i in range(len(rearrange_gates_scheduled)):
            if rearrange_gates_scheduled[i]:
                iswap_gates_scheduled[k] = rearrange_gates_scheduled[i]
                k += 1
        return iswap_gates_scheduled

    def qasm_gates_scheduled(self, iswap_gates_scheduled):
        # get qasm_gates_scheduled
        qasm_scheduled = defaultdict(list)
        k = 0
        for i in range(len(iswap_gates_scheduled)):
            layer_list = defaultdict(list)
            for j in range(len(iswap_gates_scheduled[i])):
                if isinstance(iswap_gates_scheduled[i][j][1], int):
                    layer_list['single'].append(iswap_gates_scheduled[i][j])
                else:
                    layer_list['two'].append(iswap_gates_scheduled[i][j])
            if len((layer_list['two'])) == 0 or len((layer_list['two'])) == 1:
                # At most one two-qubit gate on the same layer
                qasm_scheduled[k] = iswap_gates_scheduled[i]
                k += 1
            else:
                two_gates_list = defaultdict(list)
                for gate in layer_list['two']:
                    two_gates_list[sum(gate[1])] = gate
                sort_node = sorted(two_gates_list.keys())
                diff_node = np.diff(sort_node)
                if 4 not in diff_node:
                    # no adjacent two-qubit gates
                    qasm_scheduled[k] = iswap_gates_scheduled[i]
                    k += 1
                else:
                    subgroup_two_gates = defaultdict(list)
                    subgroup_two_gates['l'].append(two_gates_list[sort_node[0]])
                    for u in range(len(diff_node)):
                        if diff_node[u] != 4:
                            subgroup_two_gates['l'].append(two_gates_list[sort_node[u + 1]])
                        else:
                            if two_gates_list[sort_node[u]] in subgroup_two_gates['l']:
                                subgroup_two_gates['r'].append(two_gates_list[sort_node[u + 1]])
                            else:
                                subgroup_two_gates['l'].append(two_gates_list[sort_node[u + 1]])
                    for s in layer_list['single']:
                        qasm_scheduled[k].append(s)
                    for tl in subgroup_two_gates['l']:
                        qasm_scheduled[k].append(tl)
                    k += 1
                    for tr in subgroup_two_gates['r']:
                        qasm_scheduled[k].append(tr)
                    k += 1
        return qasm_scheduled

    def ScQ_gates_scheduled(self, opt_gates_list):
        # get ScQ_gates_scheduled
        gates_list = defaultdict(list)
        for i in range(len(opt_gates_list)):
            gates_list[i] = [opt_gates_list[i]]
        rearrange_gates_scheduled = copy.deepcopy(gates_list)
        for i in range(len(gates_list) - 1):
            # list_bits = [i for i in range(logical_qubits)]
            list_bits = []
            if gates_list[i]:
                if isinstance(gates_list[i][0][1], int):
                    list_bits.extend([gates_list[i][0][1]])
                if isinstance(gates_list[i][0][1], list):
                    list_bits.extend(gates_list[i][0][1])
            count = -1
            for k in range(i + 1, len(gates_list)):
                count += 1
                if gates_list[k]:
                    if isinstance(gates_list[k][0][1], int):
                        two_bits = []
                        for item in rearrange_gates_scheduled[i]:
                            if isinstance(item[1], list):
                                two_bits.extend(item[1])
                        if (gates_list[k][0][1] not in list_bits) and (gates_list[k][0][1] - 1 not in two_bits) and (
                                gates_list[k][0][1] + 1 not in two_bits):
                            rearrange_gates_scheduled[i].append(gates_list[k][0])
                            rearrange_gates_scheduled[k].remove(gates_list[k][0])
                            list_bits.extend([gates_list[k][0][1]])
                            list_bits = list(set(list_bits))
                    if isinstance(gates_list[k][0][1], list):
                        mi, ma = min(gates_list[k][0][1]), max(gates_list[k][0][1])
                        if (mi not in list_bits) and (ma not in list_bits) and (
                                mi - 1 not in list_bits) and (ma + 1 not in list_bits):
                            rearrange_gates_scheduled[i].append(gates_list[k][0])
                            rearrange_gates_scheduled[k].remove(gates_list[k][0])
                            # list_bits.extend([mi, mi - 1, ma, ma + 1])
                            list_bits.extend([mi, ma])
                            list_bits = list(set(list_bits))
                        else:
                            list_bits.extend(gates_list[k][0][1])
                            list_bits = list(set(list_bits))
                    gates_list = copy.deepcopy(rearrange_gates_scheduled)
                if (len(list_bits) == self._logical_qubits):
                    break
        k = 0
        ScQ_gates_scheduled = defaultdict(list)
        for i in range(len(rearrange_gates_scheduled)):
            if rearrange_gates_scheduled[i]:
                ScQ_gates_scheduled[k] = rearrange_gates_scheduled[i]
                k += 1
        # get ScQ_circuit
        hq = QuantumRegister(self._physical_qubits, 'q')
        ScQ_circuit = QuantumCircuit(hq)
        for i in range(len(ScQ_gates_scheduled)):
            for gate in ScQ_gates_scheduled[i]:
                if gate[0] == 'H':
                    ScQ_circuit.h(gate[1])
                if gate[0] == 'Rz':
                    ScQ_circuit.rz(gate[2], gate[1])
                if gate[0] == 'Ry':
                    ScQ_circuit.ry(gate[2], gate[1])
                if gate[0] == 'Rx':
                    ScQ_circuit.rx(gate[2], gate[1])
                if gate[0] == 'CNOT':
                    ScQ_circuit.cnot(gate[1][0], gate[1][1])
            # ScQ_circuit.barrier()
        # ScQ_circuit.measure_all()
        return ScQ_gates_scheduled, ScQ_circuit

    def qasm_gates_str(self, qasm_scheduled, logical_qubits):
        qubits = [i for i in range(logical_qubits)]
        measures = qubits
        qasm_str = str(qasm_scheduled.values())
        qasm_str = qasm_str[13:len(qasm_str) - 2]
        qasm_str = qasm_str + ',' + str(measures) + ',' + str(qubits)
        # print('qasm send:', qasm_str)
        return qasm_str

    def graph_to_qasm(self, gate='CNOT'):
        logical_circ = self.QAOA_logical_circuit()
        qubits_mapping = self.simple_layout_mapping()
        pattern_rzz_swap, rzz_gates_cycle = self.scheduled_pattern_rzz(qubits_mapping)

        physical_circ, final_gates_scheduled, final_qubits_mapping = self.QAOA_physical_circuit(pattern_rzz_swap, qubits_mapping)

        gates_list = list([])
        if gate == 'CNOT':
            hardware_gates_scheduled, hardware_circ = self.gates_decomposition(final_gates_scheduled)

            opt_hardware_gates_scheduled, optimized_circuit = self.cnot_gates_optimization(
                hardware_gates_scheduled,
                physical_qubits=self._physical_qubits)
            for layer_list in opt_hardware_gates_scheduled:
                gates_list.extend(layer_list)
        elif gate == 'iSWAP':
            gates_list, iswap_circ = self.iswap_gates_scheduled_new(final_gates_scheduled, b=4)
        else:
            print('Error: Two-qubit gate only supports CNOT and iSWAP')
        # iswap_gates_scheduled = self.iswap_finall_scheduled(gates_list, logical_qubits)
        # ScQ_gates_scheduled = self.qasm_gates_scheduled(iswap_gates_scheduled)
        ScQ_gates_scheduled, ScQ_circuit = self.ScQ_gates_scheduled(gates_list)
        # qasm_str = self.qasm_gates_str(ScQ_gates_scheduled, self._logical_qubits)
        ScQ_circuit.measure_all()
        openqasm = ScQ_circuit.qasm()
        return openqasm, final_qubits_mapping, ScQ_circuit

    def graph_sampling_energy(self, counts):
        # Obtain QAOA energy from circuit sampling results
        counts_energy = {}
        for i in range(len(counts)):
            # 0 -> -1, 1 -> 1
            result1 = [int(u) for u in list(counts[i][0])]
            # result1.reverse()
            energy1 = 0
            for node in self._nodes:
                energy1 = energy1 + node[1] * (2 * result1[node[0]] - 1)
            for edge in self._edges:
                energy1 = energy1 + edge[2] * (2 * result1[edge[0]] - 1) * (2 * result1[edge[1]] - 1)
            counts_energy[counts[i]] = energy1
        counts_energy = sorted(counts_energy.items(), key=lambda x: x[1])
        return counts_energy


    def scq_qasm(self, openqasm):
        import os
        from hardware_library import BuildLibrary
        from quafu import User
        user = User()
        user.save_apitoken(self._apitoken)
        from quafu import Task

        backend = self._cloud_backend
        task = Task()
        task.load_account()
        task.config(backend=backend)
        backend_info = task.get_backend_info()
        calibration_time = backend_info['full_info']["calibration_time"]

        logical_qubits = int(re.findall(r"\d+\.?\d*", openqasm.split('qreg')[1].split(';')[0])[0])

        if not os.path.exists('LibSubstructure_' + backend + '.txt'):
            BL = BuildLibrary(backend=backend, fidelity_threshold=96)
            BL.build_substructure_library()

        with open('LibSubstructure_' + backend + '.txt', 'r', encoding='utf-8') as f:
            substructure_data = eval(f.read())
            if substructure_data['calibration_time'] != calibration_time:
                BL = BuildLibrary(backend=backend, fidelity_threshold=96)
                BL.build_substructure_library()
        sub_clist = [qubits[0:2] for qubits in substructure_data['substructure_dict'][logical_qubits][0]]

        # # new code add linear coupling
        begin_qubit = []
        for sub in substructure_data['substructure_dict'][logical_qubits]:
            sublist = [qubits[0:2] for qubits in sub]
            G = nx.Graph()
            G.add_edges_from(sublist)
            node_degree = dict(G.degree)
            sort_degree = sorted(node_degree.items(), key=lambda kv: kv[1], reverse=True)
            if sort_degree[0][1] == 2:
                begin_qubit = sort_degree[-2][0]
                sub_clist = sublist
                break
        physical_qubits = len(substructure_data['qubit_to_int'])
        scq_qasm = openqasm.replace('qreg q[' + str(logical_qubits),
                                                  'qreg q[' + str(physical_qubits))
        qubits_list = []
        for coupling in sub_clist:
            if coupling[0] not in qubits_list:
                qubits_list.append(coupling[0])
            if coupling[1] not in qubits_list:
                qubits_list.append(coupling[1])
        qubits_list = sorted(qubits_list)
        req_to_q = {q: qubits_list[q] for q in range(len(qubits_list))}
        for req, q in sorted(req_to_q.items(), key=lambda item: item[1], reverse=True):
            scq_qasm = scq_qasm.replace('q[' + str(req) + ']', 'q[' + str(q) + ']')
        scq_qasm = scq_qasm.replace('meas[', 'c[')
        plt.close()
        return scq_qasm

    def send_quafucloud(self, scq_qasm, shots=1024):
        from quafu import QuantumCircuit as quafuQC
        from quafu import Task
        # # # send to quafu cloud
        qubits = int(re.findall(r"\d+\.?\d*", scq_qasm.split('qreg')[1].split(';')[0])[0])
        q = quafuQC(qubits)
        q.from_openqasm(scq_qasm)
        # q.draw_circuit()
        task = Task()
        task.load_account()
        task.config(backend=self._cloud_backend, shots=shots, compile=False)
        res = task.send(q)
        return res

    def run(self, shots=1024):
        from quafu import QuantumCircuit as quafuQC
        from quafu import Task

        openqasm, final_qubits_mapping, ScQ_circuit = self.graph_to_qasm(gate='CNOT')
        scq_qasm = self.scq_qasm(openqasm)
        # res = self.send_quafucloud(self, scq_qasm, shots=shots)
        # # # send to quafu cloud
        qubits = int(re.findall(r"\d+\.?\d*", scq_qasm.split('qreg')[1].split(';')[0])[0])
        q = quafuQC(qubits)
        q.from_openqasm(scq_qasm)
        # q.draw_circuit()
        task = Task()
        task.load_account()
        task.config(backend=self._cloud_backend, shots=shots, compile=False)
        res = task.send(q)
        return res

    def results(self, res):
        counts_ScQ = res.res
        counts_ScQ = sorted(counts_ScQ.items(), key=lambda x: x[1], reverse=True)
        counts_ScQ = [(item[0], item[1]) for item in counts_ScQ]

        for elem in counts_ScQ:
            if elem[1] == 0:
                counts_ScQ.remove(elem)

        counts_energy = self.graph_sampling_energy(counts_ScQ)
        return counts_energy


    def visualization(self,counts_energy):
        # draw result
        plt.close()
        optimal_solution = counts_energy[0][0][0]
        cut_node1 = []
        cut_node2 = []
        for i in range(len(optimal_solution)):
            if optimal_solution[i] == '0':
                cut_node1.append(len(optimal_solution) - 1 - i)
            else:
                cut_node2.append(len(optimal_solution) - 1 - i)
        # new_labels = dict(map(lambda x: ((x[0], x[1]), str(x[2]['weight'])), self._g.edges(data=True)))
        pos = nx.spring_layout(self._g)
        # nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=new_labels, font_size=15)
        nx.draw_networkx(self._g, pos=pos, nodelist=cut_node1, node_size=500, node_color='c', font_size=15, width=2)
        nx.draw_networkx(self._g, pos=pos, nodelist=cut_node2, node_size=500, node_color='r', font_size=15, width=2)
        nx.draw_networkx_edges(self._g, pos, width=2, edge_color='g', arrows=False)
        # plt.axis('off')
        plt.show()

        print('Several solutions with the smallest cost:\n', counts_energy[0:7])
        print('Send results to client:')
        print('cluster 1:', sorted(cut_node1))
        print('cluster 2:', sorted(cut_node2))
        print('cost:', counts_energy[0][1])

