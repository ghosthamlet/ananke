"""
Class for missing data acyclic directed mixed graphs
"""

from .admg import ADMG


class MissingADMG(ADMG):

    def __init__(self, vertices=[], di_edges=set(), bi_edges=set(), **kwargs):
        """
        Constructor

        Naming convention for vertices: R_name --> missingness indicator, X_name --> counterfactual
        Xp_name are proxies.

        :param vertices:
        :param di_edges:
        :param bi_edges:
        :param kwargs:
        """

        # initialize vertices in ADMG
        super().__init__(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges, **kwargs)

        # missing data specific stuff
        self.counterfactuals = [v for v in self.vertices if v.startswith('X_')]
        self.indicators = [v for v in self.vertices if v.startswith('R_')]
        self.proxies = [v for v in self.vertices if v.startswith('Xp_')]

        for v in self.counterfactuals:
            if v.startswith('X_'):
                var_name = v.split('_')[1]
                self.add_diedge(v, 'Xp_' + var_name)
                self.add_diedge('R_' + var_name, 'Xp_' + var_name)


    def draw(self, direction=None):
        """
        Visualize the graph.

        :return : dot language representation of the graph.
        """
        from graphviz import Digraph

        dot = Digraph()

        # set direction from left to right if that's preferred
        if direction == 'LR':
            dot.graph_attr['rankdir'] = direction

        for v in self.vertices.values():
            dot.node(v.name, shape='square' if v.fixed else 'plaintext', height='.5', width='.5')

        for parent, child in self.di_edges:
            # special clause for SWIGs
            if child.startswith('Xp_'):
                dot.edge(parent, child, color='grey')
            else:
                dot.edge(parent, child, color='blue')
        for sib1, sib2 in self.bi_edges:
            dot.edge(sib1, sib2, dir='both', color='red')

        return dot