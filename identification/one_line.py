"""
Class for one line ID
"""

import copy
from graphs.admg import ADMG


class OneLineID:

    def __init__(self, graph, interventions, outcomes):
        """
        Constructor

        :param graph: Graph on which the query will run
        :param interventions: iterable of names of variables being intervened on
        :param outcomes: iterable of names of variables whose outcomes we are interested in
        """

        self.graph = graph
        self.interventions = [A for A in interventions]
        self.outcomes = [Y for Y in outcomes]
        self.swig = copy.deepcopy(graph)
        self.swig.fix(self.interventions)
        self.ystar = {v for v in self.swig.ancestors(self.outcomes) if not self.swig.vertices[v].fixed}
        self.Gystar = self.graph.subgraph(self.ystar)
        # dictionary mapping the fixing order for each p(D | do(V\D) )
        self.fixing_orders = {}

    def draw_swig(self, direction=None):
        """
        Draw the proper SWIG corresponding to the causal query

        :return: dot language representation of the SWIG
        """

        swig = copy.deepcopy(G)

        # add fixed vertices for each intervention
        for A in self.interventions:

            fixed_vertex_name = A.lower()
            swig.add_vertex(fixed_vertex_name)
            swig.vertices[fixed_vertex_name].fixed = True

            # delete all outgoing edges from random vertex
            # and give it to the fixed vertex
            for edge in swig.di_edges:
                if edge[0] == A:
                    swig.delete_diedge(edge[0], edge[1])
                    swig.add_diedge(fixed_vertex_name, edge[1])

            # finally, add a fake edge between interventions and fixed vertices
            # just for nicer visualization
            swig.add_diedge(A, A.lower())

        return swig.draw(direction)

    def id(self):
        """
        Run one line ID for the query

        :return: Returns True if p(Y(a)) is ID, else False
        """

        self.fixing_orders = {}
        vertices = set(self.graph.vertices)

        # check if each p(D | do(V\D) ) corresponding to districts in Gystar is ID
        for district in self.Gystar.districts():

            fixable, order = self.graph.fixable(vertices - district)

            if not fixable:
                return False

            self.fixing_orders[tuple(district)] = order

        return True

    # TODO try to reduce functional
    def functional(self):
        """
        Print the functional for identification

        :return:
        """

        if not self.id():
            return "Cannot create functional, query is not ID"

        # create and return the functional
        functional = '' if set(self.ystar) == set(self.outcomes) else '\u03A3'

        for y in self.ystar:
            if y not in self.outcomes:
                functional += y
        if len(self.ystar) > 1: functional += ' '

        for district in self.fixing_orders:
            functional += '\u03A6' + ''.join(reversed(self.fixing_orders[district])) + '(p(V);G) '
        return functional

    # TODO export intermediate CADMGs for visualization
    def export_intermediates(self):

        pass


if __name__ == '__main__':

    # ID test
    vertices = ['A', 'B', 'C', 'D', 'Y']
    di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
    bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
    G = ADMG(vertices, di_edges, bi_edges)
    one_id = OneLineID(G, ['A'], ['Y'])
    one_id.draw_swig(direction='LR').render()
    #G.draw(direction='LR').render()
    print(one_id.ystar)
    print(one_id.id())
    print(one_id.functional())

    # non ID test
    vertices = ['A', 'B', 'C', 'D', 'Y']
    di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
    bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
    G = ADMG(vertices, di_edges, bi_edges)
    one_id = OneLineID(G, ['A', 'B'], ['Y'])
    print(one_id.id())
    print(one_id.functional())
