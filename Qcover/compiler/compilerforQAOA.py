import networkx as nx
import random
import math
from collections import defaultdict
import copy
import re
import matplotlib.pyplot as plt
import os
from .hardware_library import BuildLibrary
from quafu import User, Task
from quafu import QuantumCircuit as quafuQC


class CompilerForQAOA:
    """
    compilerforQAOA is used to compile the combinatorial optimization problem weight graph
    into a quantum hardware executable circuit and send it to the quafu cloud platform for execution.
    Args:
        g (networkx.Graph): Weight graph created by networkx.
        p (int): QAOA algorithm layers
        optimal_params (list): Optimal parameters of QAOA circuit calculated by Qcover.
        apitoken (str): API token of your quafu account,
                        you can get it on the quafu quantum computing cloud platform (http://quafu.baqis.ac.cn/).
        cloud_backend (str): Currently you can choose from three quantum chips: 'ScQ-P10', 'ScQ-P20', 'ScQ-P50'
    """

    def __init__(self,
                 g: nx.Graph = None,
                 p: int = 1,
                 optimal_params: list = None,
                 apitoken: str = None,
                 cloud_backend: str = "ScQ-P10") -> None:
        self._gate = "CNOT"
        self._g = g
        self._p = p
        self._optimal_params = optimal_params
        self._apitoken = apitoken
        self._cloud_backend = cloud_backend
        self._nodes = {(k, v['weight']) for k, v in g.nodes()._nodes.items()}
        self._edges = {(u, v, g.adj[u][v]['weight']) for u, v in list(g.edges())}
        self._logical_qubits = len(g.nodes())
        self._physical_qubits = self._logical_qubits
        self._logi2phys_mapping = {}

    def random_layout_mapping(self):
        """
        Args:
            physical_qubits (int): The number of hardware qubits
            logical_qubits (int): The number of qubits in a circuit
        Returns:
            qubits_mapping (dict): Random mapping of logical qubits to physical qubits
        """
        if self._physical_qubits >= self._logical_qubits:
            random_physical_qubits = random.sample(range(0, self._physical_qubits), self._logical_qubits)
            qubits_mapping = {}
            for i in range(0, self._logical_qubits):
                qubits_mapping[random_physical_qubits[i]] = i
            return qubits_mapping
        else:
            print("Error: physical qubits must be larger than logical qubits")

    def simple_layout_mapping(self, phys_qubits_order=None):
        """
        Args:
            phys_qubits_order (list): The order of the selected physical qubits
        Returns:
            qubits_mapping (dict): Simple mapping of logical qubits to physical qubits
        """
        if self._physical_qubits >= self._logical_qubits:
            phys_qubits_order = [i for i in
                                 range(0, self._logical_qubits)] if phys_qubits_order is None else phys_qubits_order
            qubits_mapping = {}
            for i in range(0, self._logical_qubits):
                qubits_mapping[i] = phys_qubits_order[i]
            return qubits_mapping
        else:
            print("Error: physical qubits must be larger than logical qubits")

    def QAOA_logical_circuit(self):
        logical_circ = quafuQC(len(self._nodes))
        graph_edges = dict(
            [(tuple(sorted(list(self._edges)[i][0:2])), list(self._edges)[i][2]) for i in range(0, len(self._edges))])
        graph_nodes = dict(self._nodes)
        gamma, beta = self._optimal_params[:self._p], self._optimal_params[self._p:]
        for i in range(len(graph_nodes)):
            logical_circ.h(i)
        for k in range(0, self._p):
            for u, v in graph_nodes.items():
                logical_circ.rz(u, 2 * gamma[k] * v)
            for u, v in graph_edges.items():
                logical_circ.rzz(u[0], u[1], 2 * gamma[k] * v)
            for u, v in graph_nodes.items():
                logical_circ.rx(u, 2 * beta[k])
        return logical_circ

    def sorted_nodes_degree(self):
        """
        Returns:
            sort_nodes (np.array): nodes are sorted by node degree in descending order
        """
        node_degree = dict(self._g.degree)
        sort_degree = sorted(node_degree.items(), key=lambda kv: kv[1], reverse=False)
        return sort_degree

    def scheduled_pattern_rzz_swap(self, qubits_mapping):
        """
        Get the fixed execution pattern of the QAOA circuit.
        Args:
            qubits_mapping (dict): {physical qubit: logical qubit}
                        example: {0: 1, 1: 2, 2: 0}
        Returns:
            pattern_rzz_swap (dict): {k: [[(q1,q2),(p1,p2)],[] ] ...},
                                    k-th execution cycle,
                                    execute rzz/swap gate between logic qubit q1 and q2 (physics qubit p1 and p2).
                                    gates execution pattern: rzz,rzz,swap,swap,rzz,rzz,swap,swap, ...
            rzz_gates_cycle (dict): rzz gate execution in k-th cycle, {(q1,q2): k, ...}
        """
        loop = 1
        cycle = 0
        pattern_rzz_swap = defaultdict(list)
        rzz_gates_cycle = defaultdict(list)
        qubits_mapping = dict(sorted(qubits_mapping.items(), key=lambda x: x[0]))
        mapping = qubits_mapping.copy()
        m = sorted(list(mapping.keys()))
        while loop <= math.ceil(self._logical_qubits / 2):
            r1 = 0
            while r1 < self._logical_qubits - 1:
                pattern_rzz_swap[cycle].append([(mapping[m[r1]], mapping[m[r1 + 1]]), (m[r1], m[r1 + 1])])
                rzz_gates_cycle[(mapping[m[r1]], mapping[m[r1 + 1]])] = [cycle, (m[r1], m[r1 + 1])]
                r1 = r1 + 2
            cycle = cycle + 1
            r2 = 1
            while r2 < self._logical_qubits - 1:
                pattern_rzz_swap[cycle].append([(mapping[m[r2]], mapping[m[r2 + 1]]), (m[r2], m[r2 + 1])])
                rzz_gates_cycle[(mapping[m[r2]], mapping[m[r2 + 1]])] = [cycle, (m[r2], m[r2 + 1])]
                r2 = r2 + 2
            cycle = cycle + 1
            if loop == math.ceil(self._logical_qubits / 2):
                break
            else:
                s1 = 1
                while s1 < self._logical_qubits - 1:
                    pattern_rzz_swap[cycle].append([(mapping[m[s1]], mapping[m[s1 + 1]]), (m[s1], m[s1 + 1])])
                    x = mapping[m[s1]]
                    mapping[m[s1]] = mapping[m[s1 + 1]]
                    mapping[m[s1 + 1]] = x
                    s1 = s1 + 2
                cycle = cycle + 1
                s2 = 0
                while s2 < self._logical_qubits - 1:
                    pattern_rzz_swap[cycle].append([(mapping[m[s2]], mapping[m[s2 + 1]]), (m[s2], m[s2 + 1])])
                    x = mapping[m[s2]]
                    mapping[m[s2]] = mapping[m[s2 + 1]]
                    mapping[m[s2 + 1]] = x
                    s2 = s2 + 2
                cycle = cycle + 1
            loop = loop + 1
        if self._logical_qubits % 2 == 1:
            pattern_rzz_swap.pop(cycle - 1)
            for item in pattern_rzz_swap[0]:
                rzz_gates_cycle[item[0]] = [0, item[1]]
        return pattern_rzz_swap, rzz_gates_cycle

    def best_initial_mapping(self, rzz_gates_cycle, truncation=5):
        """
        # Get the best initial mapping from rzz swap gates template.
        Args:
            rzz_gates_cycle (dict): rzz gate execution in k-th cycle, {(q1,q2): k, ...}
            truncation (int):  # The larger the truncation, the more likely it is to find the
                                # optimal initial mapping, but the time cost increases.
        Returns:
            best_phys2logi_mapping: {physical qubit: logical qubit}
        """
        simple_mapping = self.simple_layout_mapping()
        sorted_nodes = self.sorted_nodes_degree()
        rzz_gates_cycle = {tuple(sorted(k)): v[0] for k, v in rzz_gates_cycle.items()}
        mapping_logi2phys_list = []

        # The map is initialized in order for nodes with node degree 0.
        # TODO: In the future it may be initialized with hardware fidelity.
        degree_zero_map = {}
        bit = 0
        for node in sorted_nodes:
            if node[1]==0:
                degree_zero_map[node[0]] = bit
                bit += 1
            else:
                break

        for i in range(0, bit):
            sorted_nodes.pop(0)

        sorted_nodes_big = sorted(sorted_nodes, key=lambda kv: kv[1], reverse=True)
        for n in range(0, len(sorted_nodes_big)):
            init_map = copy.deepcopy(degree_zero_map)
            init_map[sorted_nodes_big[0][0]] = n + bit
            mapping_logi2phys_list.append((init_map, 0))


        last_cycle = max(rzz_gates_cycle.values())

        for item in range(1, len(sorted_nodes_big)):
            node = sorted_nodes_big[item][0]
            update_mapping_list = []
            for mapping_path in mapping_logi2phys_list:
                q2p_path = mapping_path[0]
                used_physical_bits = list(q2p_path.values())
                used_logical_bits = list(q2p_path.keys())
                for phys_bit in range(self._logical_qubits):
                    deepest_cycle = mapping_path[1]
                    max_depth = 0
                    if phys_bit not in used_physical_bits:
                        for neighbor_node in self._g.neighbors(node):
                            if neighbor_node in used_logical_bits:
                                depth = rzz_gates_cycle[tuple(sorted([phys_bit, q2p_path[neighbor_node]]))]
                                if depth > max_depth:
                                    max_depth = depth
                        if max_depth < last_cycle:
                            q2p_path = copy.deepcopy(q2p_path)
                            q2p_path[node] = phys_bit
                            if max_depth > deepest_cycle:
                                deepest_cycle = max_depth
                            if deepest_cycle == 0:
                                update_mapping_list.append((q2p_path, deepest_cycle))
                            else:
                                if [depth for _, depth in update_mapping_list].count(deepest_cycle) < truncation:
                                    update_mapping_list.append((q2p_path, deepest_cycle))
            if len(update_mapping_list)>1000:
                update_mapping_list = sorted(update_mapping_list, key=lambda x: x[1], reverse=False)
                mapping_logi2phys_list = copy.deepcopy(update_mapping_list[0:1000])
            else:
                mapping_logi2phys_list = copy.deepcopy(update_mapping_list)

        mapping_logi2phys_list = sorted(mapping_logi2phys_list, key=lambda x: x[1])
        if mapping_logi2phys_list:
            best_phys2logi_mapping = {v: k for k, v in mapping_logi2phys_list[0][0].items()}
        else:
            best_phys2logi_mapping = simple_mapping
        return best_phys2logi_mapping

    def QAOA_physical_circuit(self, pattern_rzz_swap, qubits_mapping):
        """
        Get the fixed execution pattern of the QAOA circuit.
        Args:
            pattern_rzz_swap (dict): Execution pattern of rzz and swap gates
            qubits_mapping (dict): {physical bit (int): logical bit (int), ...}
        Returns:
            circuit: QAOA physical circuit #qiskit
            final_gates_scheduled (dict):
        """
        graph_edges = dict(
            [(tuple(sorted(list(self._edges)[i][0:2])), list(self._edges)[i][2]) for i in range(0, len(self._edges))])
        graph_nodes = dict(self._nodes)
        gamma, beta = self._optimal_params[:self._p], self._optimal_params[self._p:]
        mapping = qubits_mapping.copy()
        qubits_mapping_initial = qubits_mapping.copy()
        qubits_mapping_initial = dict(sorted(qubits_mapping_initial.items(), key=lambda x: x[0]))
        gates_scheduled = defaultdict(list)
        m = sorted(list(mapping.keys()))
        depth = 0
        for i in range(len(graph_nodes)):
            u = mapping[m[i]]
            gates_scheduled[depth].append(('Rz', (u, graph_nodes[u])))
        depth = depth + 1

        loop = 1
        cycle = 0
        while loop <= math.ceil(len(graph_nodes) / 2):
            r1 = 0
            for i in range(len(pattern_rzz_swap[cycle])):
                if tuple(sorted(pattern_rzz_swap[cycle][i][0])) in list(graph_edges.keys()):
                    edges_weight = graph_edges[tuple(sorted(pattern_rzz_swap[cycle][i][0]))]
                    gates_scheduled[depth + cycle].append(('Rzz', (pattern_rzz_swap[cycle][i][0], edges_weight)))
                    graph_edges.pop(tuple(sorted(pattern_rzz_swap[cycle][i][0])))
                r1 = r1 + 2
            cycle = cycle + 1
            if cycle == len(pattern_rzz_swap) or not graph_edges:
                break

            r2 = 1
            for i in range(len(pattern_rzz_swap[cycle])):
                if tuple(sorted(pattern_rzz_swap[cycle][i][0])) in list(graph_edges.keys()):
                    edges_weight = graph_edges[tuple(sorted(pattern_rzz_swap[cycle][i][0]))]
                    gates_scheduled[depth + cycle].append(('Rzz', (pattern_rzz_swap[cycle][i][0], edges_weight)))
                    graph_edges.pop(tuple(sorted(pattern_rzz_swap[cycle][i][0])))
                r2 = r2 + 2
            cycle = cycle + 1
            if cycle == len(pattern_rzz_swap) or not graph_edges:
                break

            s1 = 1
            for i in range(len(pattern_rzz_swap[cycle])):
                gates_scheduled[depth + cycle].append(('SWAP', pattern_rzz_swap[cycle][i][0]))
                x = mapping[m[s1]]
                mapping[m[s1]] = mapping[m[s1 + 1]]
                mapping[m[s1 + 1]] = x
                s1 = s1 + 2
            cycle = cycle + 1

            s2 = 0
            for i in range(len(pattern_rzz_swap[cycle])):
                gates_scheduled[depth + cycle].append(('SWAP', pattern_rzz_swap[cycle][i][0]))
                x = mapping[m[s2]]
                mapping[m[s2]] = mapping[m[s2 + 1]]
                mapping[m[s2 + 1]] = x
                s2 = s2 + 2
            cycle = cycle + 1
            loop = loop + 1

        depth = depth + cycle
        for i in range(len(graph_nodes)):
            gates_scheduled[depth].append(('Rx', (mapping[m[i]], 1)))

        first_gates_scheduled = defaultdict(list)
        layer_keys = sorted(list(gates_scheduled.keys()))
        for i in range(len(layer_keys)):
            first_gates_scheduled[i] = gates_scheduled[layer_keys[i]]

        # first_gates_scheduled: The first layer of circuit
        # final_gates_scheduled: Construct the entire QAOA circuit through the first layer of circuit
        final_gates_scheduled = defaultdict(list)
        a = len(first_gates_scheduled)
        depth = 0
        for i in range(len(graph_nodes)):
            mh = list(qubits_mapping_initial.keys())
            final_gates_scheduled[depth].append(('H', mh[i], qubits_mapping_initial[mh[i]]))
        for k in range(0, self._p):
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
                        u = {v: k for k, v in qubits_mapping_initial.items()}[u]
                        if nodes_weight != 0:
                            final_gates_scheduled[depth].append(
                                ('Rz', (u, 2 * gamma[k] * nodes_weight),
                                 (qubits_mapping_initial[u], 2 * gamma[k] * nodes_weight)))

                    if first_gates_scheduled[i][j][0] == 'Rzz':
                        u, v = first_gates_scheduled[i][j][1][0]
                        edges_weight = first_gates_scheduled[i][j][1][1]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[u]
                        v = {v: k for k, v in qubits_mapping_initial.items()}[v]
                        if edges_weight != 0:
                            final_gates_scheduled[depth].append(
                                ('Rzz', ((u, v), 2 * gamma[k] * edges_weight),
                                 ((qubits_mapping_initial[u], qubits_mapping_initial[v]),
                                  2 * gamma[k] * edges_weight)))

                    if first_gates_scheduled[i][j][0] == 'SWAP':
                        u, v = first_gates_scheduled[i][j][1]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[u]
                        v = {v: k for k, v in qubits_mapping_initial.items()}[v]
                        final_gates_scheduled[depth].append(
                            ('SWAP', (u, v),
                             (qubits_mapping_initial[u], qubits_mapping_initial[v])))
                        x = qubits_mapping_initial[v]
                        qubits_mapping_initial[v] = qubits_mapping_initial[u]
                        qubits_mapping_initial[u] = x

                    if first_gates_scheduled[i][j][0] == 'Rx':
                        u = first_gates_scheduled[i][j][1][0]
                        u = {v: k for k, v in qubits_mapping_initial.items()}[u]
                        final_gates_scheduled[depth].append(
                            ('Rx', (u, 2 * beta[k]), (qubits_mapping_initial[u], 2 * beta[k])))

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
                elif final_gates_scheduled[i][0][0] == 'Rz' or final_gates_scheduled[i][0][0] == 'Rx':
                    list_bits = [final_gates_scheduled[i][k][1][0] for k in range(len(final_gates_scheduled[i]))]
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
                    elif final_gates_scheduled[k][0][0] == 'Rz' or final_gates_scheduled[k][0][0] == 'Rx':
                        for j in range(len(final_gates_scheduled[k])):
                            bits = final_gates_scheduled[k][j][1][0]
                            if bits not in list_bits:
                                rearrange_gates_scheduled[i].append(final_gates_scheduled[k][j])
                                rearrange_gates_scheduled[k].remove(final_gates_scheduled[k][j])
                                list_bits.extend((bits, bits))
                                list_bits = list(set(list_bits))
                            else:
                                pass
                    else:
                        pass
                    final_gates_scheduled = copy.deepcopy(rearrange_gates_scheduled)
                if len(list_bits) >= len(final_gates_scheduled[0]):
                    break
        k = 0
        final_gates_scheduled = defaultdict(list)
        for i in range(len(rearrange_gates_scheduled)):
            if rearrange_gates_scheduled[i]:
                final_gates_scheduled[k] = rearrange_gates_scheduled[i]
                k = k + 1

        return final_gates_scheduled, qubits_mapping_initial

    def gates_decomposition(self, final_gates_scheduled):
        # Decompose RZZ gates and SWAP gates into CNOT and single-qubit gates
        hardware_gates_scheduled = list([])
        for i in range(len(final_gates_scheduled)):
            layer_list = list([])
            layer_list1 = list([])
            layer_list2 = list([])
            for j in range(len(final_gates_scheduled[i])):
                if final_gates_scheduled[i][j][0] == 'H':
                    u = final_gates_scheduled[i][j][1]
                    layer_list.append(['H', u, 0])
                if final_gates_scheduled[i][j][0] == 'Rz':
                    u, theta = final_gates_scheduled[i][j][1]
                    layer_list.append(['Rz', u, theta])
                if final_gates_scheduled[i][j][0] == 'Rx':
                    u, theta = final_gates_scheduled[i][j][1]
                    layer_list.append(['Rx', u, theta])
                if final_gates_scheduled[i][j][0] == 'Rzz':
                    for k in range(3):
                        if k == 0:
                            u, v = final_gates_scheduled[i][j][1][0]
                            layer_list.append(['CNOT', [u, v]])
                        if k == 1:
                            u, v = final_gates_scheduled[i][j][1][0]
                            theta = final_gates_scheduled[i][j][1][1]
                            layer_list1.append(['Rz', v, theta])
                        if k == 2:
                            u, v = final_gates_scheduled[i][j][1][0]
                            layer_list2.append(['CNOT', [u, v]])
                if final_gates_scheduled[i][j][0] == 'SWAP':
                    for k in range(3):
                        if k == 0:
                            u, v = final_gates_scheduled[i][j][1]
                            ma, mi = max(u, v), min(u, v)
                            layer_list.append(['CNOT', [mi, ma]])
                        if k == 1:
                            u, v = final_gates_scheduled[i][j][1]
                            ma, mi = max(u, v), min(u, v)
                            layer_list1.append(['CNOT', [ma, mi]])
                        if k == 2:
                            u, v = final_gates_scheduled[i][j][1]
                            ma, mi = max(u, v), min(u, v)
                            layer_list2.append(['CNOT', [mi, ma]])
            if layer_list:
                hardware_gates_scheduled.append(layer_list)
            if layer_list1:
                hardware_gates_scheduled.append(layer_list1)
            if layer_list2:
                hardware_gates_scheduled.append(layer_list2)
        return hardware_gates_scheduled

    def cnot_gates_optimization(self, hardware_gates_scheduled, physical_qubits=None):
        # CNOT gates optimization
        # Two identical CNOT gates adjacent to each other are eliminated: CNOT(i,j)CNOT(i,j) = identity matrix
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
            optimized_circuit = quafuQC(physical_qubits)
            for i in range(len(opt_hardware_gates_scheduled)):
                for j in range(len(opt_hardware_gates_scheduled[i])):
                    if opt_hardware_gates_scheduled[i][j][0] == 'H':
                        optimized_circuit.h(opt_hardware_gates_scheduled[i][j][1])
                    if opt_hardware_gates_scheduled[i][j][0] == 'Rz':
                        _, v, theta = opt_hardware_gates_scheduled[i][j]
                        optimized_circuit.rz(v, theta)
                    if opt_hardware_gates_scheduled[i][j][0] == 'Rx':
                        _, v, theta = opt_hardware_gates_scheduled[i][j]
                        optimized_circuit.rx(v, theta)
                    if opt_hardware_gates_scheduled[i][j][0] == 'CNOT':
                        _, q = opt_hardware_gates_scheduled[i][j]
                        optimized_circuit.cnot(q[0], q[1])
        else:
            optimized_circuit = None
        return opt_hardware_gates_scheduled, optimized_circuit

    def graph_to_qasm(self):
        # Convert weight graph to openqasm circuit.
        qubits_mapping = self.simple_layout_mapping()
        pattern_rzz_swap, rzz_gates_cycle = self.scheduled_pattern_rzz_swap(qubits_mapping)
        best_phys2logi_mapping = self.best_initial_mapping(rzz_gates_cycle)
        pattern_rzz_swap_new = defaultdict(list)
        for k, v in pattern_rzz_swap.items():
            for gate in v:
                pattern_rzz_swap_new[k].append(
                    [(best_phys2logi_mapping[gate[0][0]], best_phys2logi_mapping[gate[0][1]]), gate[1]])

        final_gates_scheduled, final_phys2logi_mapping = self.QAOA_physical_circuit(pattern_rzz_swap_new,
                                                                                    best_phys2logi_mapping)
        hardware_gates_scheduled = self.gates_decomposition(final_gates_scheduled)
        opt_hardware_gates_scheduled, ScQ_circuit = self.cnot_gates_optimization(
            hardware_gates_scheduled, physical_qubits=self._physical_qubits)
        ScQ_circuit.measure([i for i in range(ScQ_circuit.num)], [i for i in range(ScQ_circuit.num)])
        openqasm = ScQ_circuit.to_openqasm()
        return openqasm, final_phys2logi_mapping, ScQ_circuit

    def scq_qasm(self, openqasm):
        # Compile openqasm into scq_qasm executed by quafu quantum chips.
        user = User()
        user.save_apitoken(self._apitoken)
        backend = self._cloud_backend
        task = Task()
        task.config(backend=backend)
        backend_info = task.get_backend_info()

        plt.close()
        calibration_time = backend_info['full_info']["calibration_time"]
        logical_qubits = int(re.findall(r"\d+\.?\d*", openqasm.split('qreg')[1].split(';')[0])[0])

        build_library = BuildLibrary(backend=backend, fidelity_threshold=96)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(dir_path, "backend_library")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_path = os.path.join(folder_path, 'LibSubstructure_' + backend + '.txt')

        if not os.path.exists(file_path):
            print(
                "The subgraph library of " + backend + " quantum chip does not exist. Please wait: creating subgraph library!")
            substructure_data = build_library.build_substructure_library()
            print('Complete building!')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                substructure_data = eval(f.read())
                if substructure_data['calibration_time'] != calibration_time:
                    print(
                        "The qubits of " + backend + " quantum chip have been recalibrated. Please wait: updating the subgraph library of the corresponding quantum chip!")
                    build_library.build_substructure_library()
                    print('Complete building!')
                else:
                    print(
                        "The information of qubits are unchanged, and the existing subgraph library is directly called!")

        physical_qubits = len(substructure_data['qubit_to_int'])
        scq_qasm = openqasm.replace('qreg q[' + str(logical_qubits),
                                    'qreg q[' + str(physical_qubits))

        if backend == "ScQ-P136":
            # scq_qasm = re.sub(r'barrier.*\n', "", scq_qasm)
            file_path = os.path.join(folder_path, 'LibSubchain_' + backend + '.txt')
            if not os.path.exists(file_path):
                print(
                    "The one-dimensional chain library of " + backend + " quantum chip does not exist, and is being created!")
                chain_data = build_library.build_chains_from_longest(substructure_data)
                print('Complete building!')
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chain_data = eval(f.read())
                    if chain_data['calibration_time'] != calibration_time:
                        print(
                            "Waiting: Building a library of one-dimensional chain structures for the " + backend + " quantum chip!")
                        chain_data = build_library.build_chains_from_longest(substructure_data)
                        print('Complete building!')

            longset_chain = max(chain_data['subchain_dict'].keys())
            if logical_qubits > longset_chain:
                raise SystemExit(
                    "Currently, " + backend + " quantum chip supports a maximum of " + str(longset_chain) + " qubits!")

            best_chain = chain_data['subchain_dict'][logical_qubits][0]
            nodes = []
            edges = []
            for k in best_chain:
                if k[0] not in nodes:
                    nodes.append(k[0])
                if k[1] not in nodes:
                    nodes.append(k[1])
                edges.append(k[0:2])

            G2 = nx.DiGraph()
            G2.add_nodes_from(nodes)
            G2.add_edges_from(edges)

            node_degree = dict(G2.degree)
            sort_degree = sorted(node_degree.items(), key=lambda kv: kv[1], reverse=False)
            qubits_list = []
            begin_node = sort_degree[0][0]
            qubits_list.append(begin_node)
            for k in range(len(sort_degree)):
                neighbor_nodes = [k for k, v in G2[begin_node].items()]
                neighbor_nodes = [node for node in neighbor_nodes if node not in qubits_list]
                if neighbor_nodes:
                    qubits_list.append(neighbor_nodes[0])
                    begin_node = neighbor_nodes[0]
                else:
                    break

        else:
            longset_chain = max(substructure_data['substructure_dict'].keys())
            if logical_qubits > longset_chain:
                raise SystemExit(
                    "Currently, " + backend + " quantum chip supports a maximum of " + str(longset_chain) + " qubits!")
            sub_clist = [qubits[0:2] for qubits in substructure_data['substructure_dict'][logical_qubits][0]]
            qubits_list = []
            for coupling in sub_clist:
                if coupling[0] not in qubits_list:
                    qubits_list.append(coupling[0])
                if coupling[1] not in qubits_list:
                    qubits_list.append(coupling[1])
            qubits_list = sorted(qubits_list)

        print('Physical qubits used:\n', qubits_list)

        old_qubits = []
        new_qubits = []
        req_to_q = {q: qubits_list[q] for q in range(len(qubits_list))}
        for req, q in sorted(req_to_q.items(), key=lambda item: item[1], reverse=True):
            old_qubits.append(req)
            new_qubits.append(q)
        old_qubit_pattern = r'q\[(\d+)\]'
        new_qubit_template = r'q[{}]'

        # Put the 'q[int]' number in a capturing group and replace with the contents of the capturing group
        def replace_qubits(match):
            old_qubit = int(match.group(1))
            if old_qubit in old_qubits:
                new_qubit = new_qubits[old_qubits.index(old_qubit)]
                return new_qubit_template.format(new_qubit)
            else:
                return match.group(0)

        # Replace the 'q[int]' number in the string with a new number
        scq_qasm = re.sub(old_qubit_pattern, replace_qubits, scq_qasm)
        scq_qasm = scq_qasm.replace('meas[', 'c[')
        plt.close()
        return scq_qasm

    def send(self, wait=True, shots=1024, task_name: str = '', priority: int = 2):
        """
        Send the task to the quafu cloud platform.
        Args:
            wait (bool): If you choose wait=Ture, you have to wait for the result
                         to return, during which the program will not terminate.
                         If you choose wait=False, you can submit tasks asynchronously
                         without waiting online for the results to return.
            shots (int): The number of sampling of quantum computer.
            task_name (str): The name of the task so that you can query the task status
                             on the quafu cloud platform later.
        Returns:
            task_id (str): The ID number of the task, which can uniquely identify the task.
        """
        openqasm, final_phys2logi_mapping, ScQ_circuit = self.graph_to_qasm()
        print('The depth of compiled circuit: ', len(ScQ_circuit.layered_circuit()[0]))
        print('The number of CNOT gates: ', openqasm.count('cx'))
        print('The number of single-qubit gates (Rx,Rz,H): ', openqasm.count('h') +
              openqasm.count('rx') + openqasm.count('rz'))
        self._logi2phys_mapping = {v: k for k, v in final_phys2logi_mapping.items()}
        scq_qasm = self.scq_qasm(openqasm)
        # Send to quafu cloud
        qubits = int(re.findall(r"\d+\.?\d*", scq_qasm.split('qreg')[1].split(';')[0])[0])
        q = quafuQC(qubits)
        q.from_openqasm(scq_qasm)
        task = Task()
        print(scq_qasm)
        task.config(backend=self._cloud_backend, shots=shots, compile=False, priority=priority)
        # res = task.send(q, wait=wait, name=task_name, group=task_name)
        task_id = task.send(q, wait=wait, name=task_name, group=task_name).taskid
        print("The task has been submitted to the quafu cloud platform.\nThe task ID is '%s'" % task_id)
        return task_id

    def task_status_query(self, task_id: str):
        task = Task()
        task_status = task.retrieve(task_id).task_status
        if task_status == 'In Queue' or task_status == 'Running':
            print("The current task status is '%s', please wait." % task_status)
        elif task_status == 'Completed':
            print("The task execution has completed and the result has been returned.")
            res = task.retrieve(task_id)
            return res

    def right_left_counts_rearrange(self, logi2phys_mapping, counts):
        # The bits are arranged as 0, 1, 2,... from right to the left.
        qubit_str = list(counts.keys())
        counts_new = defaultdict(list)
        for i in range(len(qubit_str)):
            str = ''
            for k in range(len(logi2phys_mapping)):
                str = str + qubit_str[i][::-1][logi2phys_mapping[k]]
            counts_new[str[::-1]] = counts[qubit_str[i]]
        return counts_new

    def left_right_counts_rearrange(self, logi2phys_mapping, counts):
        # The bits are arranged as 0, 1, 2,... from left to the right.
        qubit_str = list(counts.keys())
        counts_new = defaultdict(list)
        for i in range(len(qubit_str)):
            str = ''
            for k in range(len(logi2phys_mapping)):
                str = str + qubit_str[i][::][logi2phys_mapping[k]]
            counts_new[str] = counts[qubit_str[i]]
        return counts_new

    def graph_sampling_energy_ising(self, counts):
        # Obtain QAOA energy from circuit sampling results
        counts_energy = {}
        for i in range(len(counts)):
            # 0 -> -1, 1 -> 1
            result1 = [int(u) for u in list(counts[i][0])]
            energy1 = 0
            for node in self._nodes:
                energy1 = energy1 + node[1] * (2 * result1[node[0]] - 1)
            for edge in self._edges:
                energy1 = energy1 + 2 * edge[2] * (2 * result1[edge[0]] - 1) * (2 * result1[edge[1]] - 1)
            counts_energy[counts[i]] = energy1
        counts_energy = sorted(counts_energy.items(), key=lambda x: x[1])
        return counts_energy

    def graph_sampling_energy_qubo(self, counts):
        # Obtain QUBO cost from circuit sampling results.
        counts_energy = {}
        import numpy as np
        qubo_mat = np.zeros([len(self._nodes), len(self._nodes)])
        ising_mat = np.zeros([len(self._nodes), len(self._nodes)])
        for edge in self._edges:
            ising_mat[edge[0], edge[1]] = edge[2]
            ising_mat[edge[1], edge[0]] = edge[2]
            qubo_mat[edge[0], edge[1]] = 4 * edge[2] / 2.
            qubo_mat[edge[1], edge[0]] = 4 * edge[2] / 2.
        for node in self._nodes:
            ising_mat[node[0], node[0]] = node[1]
            qubo_mat[node[0], node[0]] = (node[1] - sum(2. * qubo_mat[node[0]] / 4.)) * 2
        for i in range(len(counts)):
            result1 = np.array([int(u) for u in list(counts[i][0])])
            energy1 = np.dot(np.dot(result1, qubo_mat), result1)
            counts_energy[counts[i]] = energy1
        counts_energy = sorted(counts_energy.items(), key=lambda x: x[1])
        return counts_energy

    def results_processing(self, results):
        # The sampling result is the distribution of hardware physical qubit strings,
        # and the physical qubits need to be mapped back to the weight graph nodes.
        counts_ScQ0 = results.res
        logi2phys_mapping = self._logi2phys_mapping
        counts_ScQ_new = self.left_right_counts_rearrange(logi2phys_mapping, counts_ScQ0)
        counts_ScQ = sorted(counts_ScQ_new.items(), key=lambda x: x[1], reverse=True)
        counts_ScQ = [(item[0], item[1]) for item in counts_ScQ]

        for elem in counts_ScQ:
            if elem[1] == 0:
                counts_ScQ.remove(elem)

        counts_energy = self.graph_sampling_energy_qubo(counts_ScQ)
        print('Results ((qubits str, number of sampling), QUBO Cost):\n', counts_energy)
        return counts_energy

    def visualization(self, counts_energy, problem='MaxCut', solutions=3, problem_graph=None):
        """
            Visualize optimal solutions to combinatorial optimization problems
            based on sampling results from quafu hardware.
        Args:
            counts_energy: self.results_processing()
            problem: Visualization of two types of problems is currently supported, 'MaxCut' or 'GraphColoring'
            solutions: Visualize the top "solutions" optimal solutions
            problem_graph: original problem graph
        """
        plt.close()
        print('Send results to client:')
        if problem == 'MaxCut':
            for s in range(solutions):
                optimal_solution = counts_energy[s][0][0]
                cut_node1 = []
                cut_node2 = []
                for i in range(len(optimal_solution)):
                    if optimal_solution[i] == '0':
                        cut_node1.append(i)
                    else:
                        cut_node2.append(i)
                pos = nx.spring_layout(self._g)
                # pos = nx.circular_layout(self._g)
                nx.draw_networkx(self._g, pos=pos, nodelist=cut_node1, node_size=500, node_color='c', font_size=15,
                                 width=2)
                nx.draw_networkx(self._g, pos=pos, nodelist=cut_node2, node_size=500, node_color='r', font_size=15,
                                 width=2)
                nx.draw_networkx_edges(self._g, pos, width=2, edge_color='g', arrows=False)
                # plt.axis('off')
                plt.tight_layout()
                plt.show()
                print('======== Solution ' + str(1 + s) + ' ========')
                print('Cluster 1:', sorted(cut_node1))
                print('Cluster 2:', sorted(cut_node2))
                print('Cost (QUBO):', counts_energy[s][1])
        elif problem == 'GraphColoring':
            import matplotlib.colors
            color_list = ['r', 'c', 'b', 'y', 'm']
            color_list = color_list + list(matplotlib.colors.cnames.values())
            nodes = len(problem_graph.nodes)
            color_num = int(len(counts_energy[0][0][0]) / nodes)
            for s in range(solutions):
                print('======== Solution ' + str(1 + s) + ' ========')
                optimal_solution = counts_energy[s][0][0]
                qubo_bits = []
                for n in range(0, nodes):
                    qubo_bits.append(optimal_solution[n * color_num:(n + 1) * color_num])
                color_dic = {}
                for k in range(nodes):
                    color_dic.setdefault(qubo_bits[k], []).append(k)
                pos = nx.spring_layout(problem_graph)
                # pos = nx.circular_layout(self._g)
                color = 0
                for bit, node_list in color_dic.items():
                    nx.draw_networkx(problem_graph, pos=pos, nodelist=node_list, node_size=500,
                                     node_color=color_list[color], font_size=15, width=2)
                    color = color + 1
                    print('Coloring ' + str(color) + ':', sorted(node_list))
                nx.draw_networkx_edges(problem_graph, pos, width=2, edge_color='g', arrows=False)
                # plt.axis('off')
                plt.tight_layout()
                plt.show()
                print('Cost (QUBO):', counts_energy[s][1])
        else:
            print('The current version does not support visualization of this type of problem!')
