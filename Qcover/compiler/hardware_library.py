import networkx as nx
from queue import PriorityQueue
import matplotlib.pyplot as plt
import copy
import math
from collections import defaultdict
import os
import re
from quafu import Task
import numpy as np


class BuildLibrary:
    def __init__(self, backend='ScQ-P10', fidelity_threshold=95):
        self.backend = backend
        self.fidelity_threshold = fidelity_threshold
        task = Task()
        task.load_account()
        task.config(backend=self.backend)
        self.backend_info = task.get_backend_info()
        self.calibration_time = self.backend_info['full_info']["calibration_time"]

    def get_structure(self):
        """
        Args:
            backend (str): topology data file. 'structure_ScQ-P20.txt'
        Returns:
            int_to_qubit(dict): {0: Q01, 1: Q02, ...}
            qubit_to_int(dict): {Q01: 0, Q02: 0, ...}
            directed_weighted_edges(list):[[qubit1,qubit2,fidelity],...]
            connected_substructure_list(list): [networkx.Graph,...]
        """

        json_topo_struct = self.backend_info['full_info']["topological_structure"]

        # # not ordered
        # qubits_list = []
        # for gate in json_topo_struct.keys():
        #     qubit = gate.split('_')
        #     if qubit[0] not in qubits_list:
        #         qubits_list.append(qubit[0])
        #     if qubit[1] not in qubits_list:
        #         qubits_list.append(qubit[1])

        # # ordered
        qubits_list = []
        for gate in json_topo_struct.keys():
            qubit = gate.split('_')
            qubits_list.append(qubit[0])
            qubits_list.append(qubit[1])
        qubits_list = list(set(qubits_list))
        qubits_list = sorted(qubits_list, key=lambda x: int(re.findall(r"\d+", x)[0]))

        int_to_qubit = {k: v for k, v in enumerate(qubits_list)}
        qubit_to_int = {v: k for k, v in enumerate(qubits_list)}
        directed_weighted_edges = []
        weighted_edges = []
        edges_dict = {}
        for gate, name_fidelity in json_topo_struct.items():
            gate_qubit = gate.split('_')
            qubit1 = qubit_to_int[gate_qubit[0]]
            qubit2 = qubit_to_int[gate_qubit[1]]
            gate_name = list(name_fidelity.keys())[0]
            fidelity = name_fidelity[gate_name]['fidelity']
            directed_weighted_edges.append([qubit1, qubit2, fidelity])
            gate_reverse = gate.split('_')[1] + '_' + gate.split('_')[0]
            if gate not in edges_dict and gate_reverse not in edges_dict:
                edges_dict[gate] = fidelity
            else:
                if fidelity < edges_dict[gate_reverse]:
                    edges_dict.pop(gate_reverse)
                    edges_dict[gate] = fidelity

        for gate, fidelity in edges_dict.items():
            gate_qubit = gate.split('_')
            qubit1, qubit2 = qubit_to_int[gate_qubit[0]], qubit_to_int[gate_qubit[1]]
            weighted_edges.append([qubit1, qubit2, fidelity])

        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edges)
        G0 = copy.deepcopy(G)
        connected_substructure_list = []
        while len(G.nodes()) > 0:
            connected_nodes = max(nx.connected_components(G), key=len)
            connected_subgraph = G0.subgraph(connected_nodes)
            connected_substructure_list.append(connected_subgraph)
            G.remove_nodes_from(connected_nodes)
            # # draw coupling graph
            plt.close()
            # pos = nx.spring_layout(connected_subgraph)
            # new_labels = dict(map(lambda x: ((x[0], x[1]), format(float(str(x[2]['weight'])), '.2f')),
            #                       connected_subgraph.edges(data=True)))
            # nx.draw_networkx_edge_labels(connected_subgraph, pos=pos, edge_labels=new_labels, font_size=8)
            # nx.draw(connected_subgraph, pos=pos, with_labels=True, node_color='r', node_size=150, edge_color='b',
            #         width=1, font_size=9)
            # plt.show()
        return int_to_qubit, qubit_to_int, directed_weighted_edges, connected_substructure_list

    def substructure(self, directed_weighted_edges, connected_substructure_list, qubits_need):
        all_substructure = []
        substructure_nodes = []
        while len(all_substructure) == 0:
            for cg in connected_substructure_list:
                if len(cg.nodes()) >= qubits_need:
                    sorted_edges = sorted(cg.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
                    for elem in sorted_edges:
                        if elem[2]['weight'] > self.fidelity_threshold:
                            neighbors = PriorityQueue()
                            neighbors.put((-1, elem[0]))
                            ret_nodes = []
                            log_weight_product = 0
                            for node in cg.nodes():
                                cg.nodes[node]['visited'] = False
                            while neighbors.not_empty:
                                temp = neighbors.get()
                                node = temp[1]
                                if cg.nodes[node]['visited']:
                                    continue
                                weight = -temp[0]
                                log_weight_product += math.log(weight)
                                cg.nodes[node]['visited'] = True
                                ret_nodes.append(node)
                                if len(ret_nodes) == qubits_need:
                                    break
                                for neighbor in cg[node]:
                                    if not cg.nodes[neighbor]['visited']:
                                        weight = cg[node][neighbor]['weight']
                                        neighbors.put((-weight, neighbor))
                            out = []
                            for edge in directed_weighted_edges:
                                if edge[0] in ret_nodes and edge[1] in ret_nodes:
                                    out.append(edge)
                            if sorted(ret_nodes) not in substructure_nodes and all(
                                    qubit[2] > self.fidelity_threshold for qubit in out):
                                substructure_nodes.append(sorted(ret_nodes))
                                all_substructure.append([log_weight_product, out])

            self.fidelity_threshold = self.fidelity_threshold - 1
        all_substructure = sorted(all_substructure, key=lambda x: x[0], reverse=True)
        return all_substructure

    def build_substructure_library(self):
        # substructure_dict = defaultdict(list)
        substructure_dict = {}
        int_to_qubit, qubit_to_int, directed_weighted_edges, connected_substructure_list = self.get_structure()
        substructure_dict[1] = [[[q, q, 0.99]] for q in int_to_qubit.keys()]
        for qubits in range(2, len(connected_substructure_list[0].nodes()) + 1):
            sub_graph = self.substructure(directed_weighted_edges, connected_substructure_list, qubits)
            qlist = []
            for j in range(len(sub_graph)):
                # substructure_dict[qubits].append(sub_graph[j][1])
                qlist.append(sub_graph[j][1])
            substructure_dict[qubits] = qlist

        sorted_weighted_edges = sorted(directed_weighted_edges, key=lambda x: x[2], reverse=True)
        save_substructure = {'calibration_time': self.calibration_time, 'structure': sorted_weighted_edges, 'substructure_dict': substructure_dict,
                             'int_to_qubit': int_to_qubit, 'qubit_to_int': qubit_to_int}
        if os.path.exists('LibSubstructure_' + self.backend + '.txt'):
            os.remove('LibSubstructure_' + self.backend + '.txt')
        with open('LibSubstructure_' + self.backend + '.txt', 'w') as file:
            # file.write(json.dumps(save_substructure))
            file.write(str(save_substructure))
        return save_substructure

    def chain_library_2D(self, substructure_data):
        # find one-dimensional chain
        G = nx.DiGraph()
        G.add_weighted_edges_from(substructure_data['substructure_dict'][len(substructure_data['qubit_to_int'])][0])
        node_degree = dict(G.degree)
        sort_degree = sorted(node_degree.items(), key=lambda kv: kv[1], reverse=False)
        one_link_nodes = [node for node, degree in sort_degree if degree == 2]
        all_oneD_chain = []
        for i in range(len(one_link_nodes)):
            begin_node = one_link_nodes[i]
            queue_oneD_chain = []
            queue_oneD_chain.append([begin_node])

            while queue_oneD_chain:
                end_node = queue_oneD_chain[0][-1]
                neighbor_nodes = [k for k, v in G[end_node].items()]
                neighbor_nodes = [node for node in neighbor_nodes if node not in queue_oneD_chain[0]]
                if len(neighbor_nodes) == 0:
                    all_oneD_chain.append(queue_oneD_chain[0])
                    queue_oneD_chain.remove(queue_oneD_chain[0])
                elif len(neighbor_nodes) == 1:
                    queue_oneD_chain[0].append(neighbor_nodes[0])
                else:
                    for k in range(1, len(neighbor_nodes)):
                        oneD_chain = copy.deepcopy(queue_oneD_chain[0])
                        oneD_chain.append(neighbor_nodes[k])
                        if oneD_chain not in queue_oneD_chain:
                            queue_oneD_chain.append(oneD_chain)
                    queue_oneD_chain[0].append(neighbor_nodes[0])

        chain_dict = defaultdict(list)

        sorted_chain = []
        for chain in all_oneD_chain:
            if sorted(chain) not in sorted_chain:
                sorted_chain.append(sorted(chain))
                chain_dict[len(chain)].append(chain)

        chain_dict = dict(sorted(chain_dict.items(), key=lambda x: x[0]))
        # longset_chain = chain_dict[max(chain_dict.keys())][0]

        structure_dict = {}
        for edge in substructure_data['structure']:
            structure_dict[(edge[0], edge[1])] = edge[2]

        connected_chain_list = []
        for node_number, chain_nodes_list in chain_dict.items():
            for chain_nodes in chain_nodes_list:
                chain_graph = nx.DiGraph()
                directed_weighted_edges = []
                for i in range(len(chain_nodes)-1):
                    directed_weighted_edges.append([chain_nodes[i], chain_nodes[i+1],
                                                    structure_dict[(chain_nodes[i], chain_nodes[i+1])]])
                    directed_weighted_edges.append([chain_nodes[i+1], chain_nodes[i],
                                                    structure_dict[(chain_nodes[i+1], chain_nodes[i])]])
                chain_graph.add_weighted_edges_from(directed_weighted_edges)
                connected_chain_list.append([directed_weighted_edges, chain_graph])

        subchain_dict = defaultdict(list)
        for chain in connected_chain_list:
            for qubits in range(2, len(chain[1].nodes())+1):
                sub_graph = self.substructure(chain[0], [chain[1]], qubits)
                for j in range(len(sub_graph)):
                    log_weight_product = 0
                    for edge in sub_graph[j][1]:
                        log_weight_product = log_weight_product + math.log(edge[2])
                    if [log_weight_product, sub_graph[j][1]] not in subchain_dict[qubits]:
                        subchain_dict[qubits].append([log_weight_product, sub_graph[j][1]])

        sorted_subchain_dict = {}
        for k, chain_list in subchain_dict.items():
            chain_list = sorted(chain_list, key=lambda x: x[0], reverse=True)
            chain_list = [chain[1] for chain in chain_list]
            sorted_subchain_dict[k] = chain_list

        save_substructure = {'calibration_time': self.calibration_time, 'subchain_dict': sorted_subchain_dict}
        if os.path.exists('LibSubchain_' + self.backend + '.txt'):
            os.remove('LibSubchain_' + self.backend + '.txt')
        with open('LibSubchain_' + self.backend + '.txt', 'w') as file:
            file.write(str(save_substructure))
        return save_substructure

