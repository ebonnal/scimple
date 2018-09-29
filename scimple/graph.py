from abc import ABC, abstractmethod
from collections import Iterable
import numpy as np
import networkx as nx
from copy import copy
import matplotlib.pyplot as plt
from .plot import Plot, random_color_well_dispatched, pastelize
from .utils import flatten_n_times

# #####
# GRAPH
# #####


class Graph(nx.DiGraph):
    """
    Scimplify Graph creation on top of networkx library
    """
    IN = 0
    OUT = 1

    def copy(self):
        copy_ = super().copy()
        copy_._isdirected = copy(self._isdirected)
        copy_._plot = copy(self._plot)
        copy_.title = copy(self.title)
        copy_._edges_labels = copy(self._edges_labels)
        copy_._nodes_labels = copy(self._nodes_labels)
        copy_._nodes_grouped_by_label = copy(self._nodes_grouped_by_label)
        copy_._nodes = copy(self._nodes)

    def __init__(self, title="", directed=True):
        self._isdirected = directed
        self._plot = None
        self.title = title
        self._edges_labels = {}  # {(node1_edge1, node2_edge1): edge1_label, ...}
        self._nodes_labels = {}  # {node_name1: node_label1, ...}
        self._nodes_grouped_by_label = {} # {'label1': [node_name1,...],...}
        self._nodes = {} #{node_name: [{node_in1,...}, {node_out1,...}], ...}
        super().__init__()

    def change_node_label(self, name, new_label):
        self._nodes_grouped_by_label[self._nodes_labels[name]].remove(name)
        if new_label in self._nodes_grouped_by_label:
            self._nodes_grouped_by_label[new_label].append(name)
        else:
            self._nodes_grouped_by_label[new_label] = [name]
        self._nodes_labels[name] = new_label

    def change_edge_label(self, node_from, node_to, new_label):
        self._edges_labels[(node_from, node_to)] = new_label
        if not self.is_directed():
            self._edges_labels[(node_to, node_from)] = new_label

    def get_edge_label(self, node_from, node_to):
        return self._edges_labels[(node_from, node_to)]

    def get_node_label(self, name):
        return self._nodes_labels[name]

    def neigh(self, node, in_out=None):
        if in_out is None and not self._isdirected:
            return self._nodes[node][0]
        elif in_out == Graph.IN:
            return self._nodes[node][0]
        elif in_out == Graph.OUT:
            return self._nodes[node][1]
        raise ValueError("Graph.OUT or Graph.IN expected")

    def add_node(self, name, label=None):
        self._nodes[name] = [set(), set()]
        if label in self._nodes_grouped_by_label:
            self._nodes_grouped_by_label[label].append(name)
        else:
            self._nodes_grouped_by_label[label] = [name]
        self._nodes_labels[name] = label
        super().add_node(name)

    def add_edge(self, node1, node2, label=None, symetric=True):
        self._nodes[node1][1].add(node2)
        self._nodes[node2][0].add(node1)
        self._edges_labels[(node1, node2)] = label
        if not self._isdirected:
            self._nodes[node1][0].add(node2)
            self._nodes[node2][1].add(node1)
            self._edges_labels[(node2, node1)] = label
        super().add_edge(node1, node2)

    def plot(self, nodes_size=400, font_size_nodes=7, font_size_edges=6, colored_by=None, random_seed=None):
        """

        :param nodes_size: int
        :param font_size_nodes: int
        :param font_size_edges: int
        :param colored_by:
            None: colored by label
            dict: list of lists of nodes names: example: for A,B,C,D,E nodes, colored_by=[["A", "B"], ["C", "D"]] will
                  creates 3 groups: A&B, C&D and E (not mandatory to mention all nodes, a group for non mentioned nodes
                  is created
        :return:
        """
        # New colors, Plot axe and pos each time
        axe = Plot(dim=2, title=self.title).axe
        axe.axis('off')
        pos = nx.spring_layout(self, scale=2)
        # DRAW NODES
        if colored_by is not None:
            colored_by_dict = {i: colored_by[i] for i in range(len(colored_by))}
            mentioned_nodes = flatten_n_times(1, colored_by)
            if len(mentioned_nodes) != len(self.nodes):
                default_nodes_group = []
                for node in self.nodes:
                    if node not in mentioned_nodes:
                        default_nodes_group.append(node)
                colored_by_dict[None] = default_nodes_group
        else:
            colored_by_dict = self._nodes_grouped_by_label
        n_color_well_dispatched = random_color_well_dispatched(len(colored_by_dict), random_seed=random_seed)
        for i, label in enumerate(colored_by_dict):
            nx.draw_networkx_nodes(self, pos=pos, ax=axe, alpha=1, with_labels=True,
                                   nodelist=colored_by_dict[label],
                                   node_color=pastelize(n_color_well_dispatched[i]),
                                   node_size=nodes_size,
                                   font_size=font_size_nodes)
        # DRAW EDGES
        if self._isdirected:
            nx.draw_networkx_edges(self, pos=pos, ax=axe,
                                   edge_list=list(self.edges),
                                   arrowstyle='->', arrows=True, node_size=nodes_size)
        else:
            nx.draw_networkx_edges(self, pos=pos, ax=axe,
                                   edge_list=list(self.edges),
                                   arrows=False, node_size=nodes_size)
        # DRAW NODES LABELS
        nx.draw_networkx_labels(self, pos=pos, ax=axe, font_size=font_size_nodes,
                                node_size=nodes_size,
                                labels={node: str(node) +
                                ("" if self._nodes_labels[node] is None else
                                f"\n{self._nodes_labels[node]}") for node in self._nodes})
        # DRAW EDGES LABELS
        nx.draw_networkx_edge_labels(self, pos=pos, ax=axe,
                                     edge_list=list(self.edges),
                                     edge_labels=self._edges_labels,
                                     font_size=font_size_edges,
                                     font_size_nodes=0,
                                     node_size=nodes_size,
                                     label_pos=0.25 if self._isdirected else 0.5)

def dijkstra(G, start, target):
    """
    :param start: node name
    :param target: node name
    :param G: scimple.graph.Graph: non directed, with nodes labels at 0 and edges with cost
    :return: pair tuple: (list of nodes name for shortest path, Graph with colorized path)
    """
    curr_node = start
    nearest_node = start
    nearest_node_dist = 7
    for next_node in G.neigh(start):
        pass