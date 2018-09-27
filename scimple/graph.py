from abc import ABC, abstractmethod
from collections import Iterable
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .plot import Plot, random_color_well_dispatched
# #####
# GRAPH
# #####
# Functionnalities around graph algorithmic
# class Edge:
#     def __init__(self, node1, node2, label=None, oriented=True):
#         """
#         :param node1: scimple.graph.Node
#         :param node2: scimple.graph.Node
#         :param label: object
#         :param oriented: bool
#         """
#         self.node1 = node1
#         self.node2 = node2
#         self.label = label
#         self.oriented = oriented
#     def __str__(self):
#         return f"<{self.node1}->{self.node2})" if self.oriented else f"({self.node1}<->{self.node2})"
#
# class Node:
#     def __init__(self, name, label=None):
#         """
#         :param name: object
#         :param label: object
#         """
#         self.name = str(name)
#         self.label = label
#
#     def __str__(self):
#         return f"[|{self.name},'{self.label}'|]"
#
# class AbstractGraph(ABC, Iterable):
#     """
#     Abstract class for
#
#     Must implement:
#         - add_node
#         - add_edge
#         - __next__ & __iter__
#         - get_neighbours
#         - get_edges
#         - get_nodes
#     Should implement
#         - to_nx_graph in order to call 'show'
#     """
#     def add_nodes(self, nodes):
#         """
#         :param nodes: collection.Iterable of scimple.graph.Node
#         :return: None
#         """
#         for node in nodes:
#             self.add_node(node)
#
#     def add_edges(self, edges):
#         """
#         :param edges: collection.Iterable of scimple.graph.Edge
#         :return: None
#         """
#         for edge in edges:
#             self.add_edge(edge)
#
#     def __str__(self):
#         print(f"Nodes ({len(self.get_nodes())}):\n"
#               f"{[str(node) for node in self.get_nodes()]}\n"
#               f"\nEdges ({len(self.get_edges())}):\n"
#               f"{[str(edge) for edge in self.get_edges()]}\n")
#
#     @abstractmethod
#     def add_node(self, node): pass
#
#     @abstractmethod
#     def add_edge(self, edge): pass
#
#     @abstractmethod
#     def get_neighbours(self, node):
#         """
#         :param node: scimple.graph.Node
#         :return: collections.Collection of scimple.graph.Node
#         """
#
#     @abstractmethod
#     def get_nodes(self, node):
#         """
#         :param node: scimple.graph.Node
#         :return: collections.Collection of scimple.graph.Node
#         """
#
#     @abstractmethod
#     def get_edges(self, node):
#         """
#         :param node: scimple.graph.Node
#         :return: collections.Collection of scimple.graph.Node
#         """
#
#     def show(self):
#         raise NotImplementedError
#         # fig, axe = plt.subplots()
#         # nx.draw_networkx_edge_labels(self.to_nx_graph(), ax=axe,
#         #                              with_labels=True,
#         #                              edge_labels={: edge.label for edge in self.get_edges()})
#         # plt.show(block=True)
#
# class HashGraph(AbstractGraph):
#     def __init__(self):
#         self.hash_map = {}
#         super().__init__()
#     def __iter__(self):
#         for node in self.hash_map:
#             yield node
#     def add_edge(self, edge):
#         if not edge.node1 in self.hash_map:
#
#     def add_node(self, node):pass
#     def get_edges(self):pass
#     def get_nodes(self):pass
#     def get_neighbours(self, node):pass
#

class Graph(nx.DiGraph):
    """
    """
    IN = 0
    OUT = 1
    def __init__(self, title=""):
        """
        """
        self._plot = None
        self.title = title
        self._edges_labels = {}
        self._nodes_labels = {}
        self._nodes_grouped_by_label = {}
        self._nodes = {} # elems : {node_name: [{nodes_in}, {nodes_out}], ...}
        super().__init__()
    def neigh(self, node, in_out):
        if in_out == Graph.IN:
            return self._nodes[node][0]
        elif in_out == Graph.OUT:
            return self._nodes[node][0]
        raise ValueError("Graph.OUT or Graph.IN expected")
    def add_edge(self, node1, node2, label=None, symetric=True):
        self._nodes[node1][1].add(node2)
        self._nodes[node2][0].add(node1)
        if label is not None:
            self._edges_labels[(node1, node2)] = label
        super().add_edge(node1, node2)
    def add_node(self, name, label=None):
        self._nodes[name] = [set(), set()]
        if label in self._nodes_grouped_by_label:
            self._nodes_grouped_by_label[label].append(name)
        else:
            self._nodes_grouped_by_label[label] = [name]
        if label is None:
            self._nodes_labels[name] = name
        else:
            self._nodes_labels[name] = label
        super().add_node(name)
    def plot(self):
        n_color_well_dispatched = random_color_well_dispatched(len(self._nodes_grouped_by_label))
        axe = Plot(dim=2, title=self.title).axe
        axe.axis('off')
        pos = nx.spring_layout(self, scale=2)
        for i, label in enumerate(self._nodes_grouped_by_label):
            print({node: node for node in self._nodes_grouped_by_label[label]})
            nx.draw_networkx_nodes(self, pos=pos, ax=axe, alpha=1, with_labels=True,
                                   nodelist=self._nodes_grouped_by_label[label],
                                   node_color=n_color_well_dispatched[i])
        nx.draw_networkx_edges(self, pos=pos, ax=axe,
                               edge_list=list(self.edges),
                               arrowstyle='->')
        nx.draw_networkx_labels(self, pos=pos, ax=axe,
                                label={node: node for node in self._nodes_grouped_by_label[label]})

        nx.draw_networkx_edge_labels(self, pos=pos, ax=axe,
                                     edge_list=list(self.edges),
                                     edge_labels=self._edges_labels,
                                     font_size=5)

    # def _get_plot_axe(self):
    #     if self._plot is None:
    #         self._plot =
    #     return self._plot.axe

    # def __getattr__(self, item):
    #     if item.startswith("draw"):
    #         try:
    #             eval(f"nx.{item}")
    #             self._draw_method = item
    #             return self._wrapper_func
    #         except AttributeError:
    #             raise AttributeError(f"Attribute {item} is invalid")
    #     else:
    #         raise AttributeError(f"Attribute {item} is invalid")
    #
    # def _wrapper_func(self, **kwargs):
    #     invalid_kwargs = {"G", "ax", "pos", "alpha"}
    #     if any([kwarg in invalid_kwargs for kwarg in kwargs]):
    #         raise AttributeError(f"Following kwargs cannot be given: {invalid_kwargs}")
    #     kwargs["G"] = self
    #     kwargs["ax"] = self._get_plot_axe()
    #     kwargs["alpha"] = 0.6
    #     kwargs["pos"] = self._pos
    #     eval(f"nx.{self._draw_method}(**kwargs)")

