import networkx as nx

def get_graph_weights(graph):
    """
    get the weights of nodes and edges in graph
    Args:
        graph (nx.Graph): graph to get weight of nodes and edges
    Return:
        node weights form is dict{nid1: node_weight}, edges weights form is dict{(nid1, nid2): edge_weight}
    """
    nodew = nx.get_node_attributes(graph, 'weight')
    edw = nx.get_edge_attributes(graph, 'weight')
    edgew = edw.copy()
    for key, val in edw.items():
        edgew[(key[1], key[0])] = val

    return nodew, edgew