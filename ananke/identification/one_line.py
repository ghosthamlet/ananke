"""
Class for one line ID
"""

import copy
import os
from ..graphs.admg import ADMG
from itertools import chain, combinations


class NotIdentifiedError(Exception):
    """
    Custom error for when desired functional is not identified
    """
    pass


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

        swig = copy.deepcopy(self.graph)

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

            # if any piece is not ID, return not ID
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
            raise NotIdentifiedError

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
    def export_intermediates(self, folder="intermediates"):
        """
        Export intermediate CADMGs obtained during fixing

        :param folder: string specifying path to folder where the files will be written
        :return: None
        """

        # make the folder if it doesn't exist
        if not os.path.exists(folder):
            os.mkdir(folder)

        # clear the directory
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # do the fixings and render intermediate CADMGs
        for district in self.fixing_orders:

            G = copy.deepcopy(self.graph)

            fixed_vars = ""
            dis_name = "".join(district)

            for v in self.fixing_orders[district]:
                fixed_vars += v
                G.fix(v)
                G.draw().render(os.path.join(folder,
                                             "phi" + fixed_vars + "_dis" + dis_name + ".gv"))


def powerset(iterable, min_size):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable((combinations(s, r)) for r in range(min_size, len(s) + 1))


def get_intrinsic_sets(graph):
    intrinsic = set()
    vertices = set(graph.vertices)
    order_dict = dict()
    fixed_vertices = set()
    for v in graph.vertices:
        if graph.vertices[v].fixed == True:
            fixed_vertices.add(v)

    for var in vertices:
        fixable, order = graph.fixable(vertices - set(var) - fixed_vertices)

        if fixable:
            intrinsic_set = frozenset([var])
            intrinsic.add(intrinsic_set)
            order_dict[intrinsic_set] = order
    for district in graph.districts():
        # There is possibly a more efficient way of doing this
        for pset in powerset(district, 2):
            fixable, order = graph.fixable(vertices - set(pset) - fixed_vertices)
            if fixable:
                intrinsic_set = frozenset(list(pset))
                intrinsic.add(intrinsic_set)
                order_dict[intrinsic_set] = order

    return intrinsic, order_dict


class OneLineGZID:

    def __init__(self, graph, interventions, outcomes):
        self.graph = graph
        self.interventions = interventions
        self.outcomes = outcomes
        self.swig = copy.deepcopy(graph)
        self.swig.fix(self.interventions)
        self.ystar = {v for v in self.swig.ancestors(self.outcomes) if not self.swig.vertices[v].fixed}
        self.Gystar = self.graph.subgraph(self.ystar)

    def _required_intrinsic_sets(self):
        required_intrinsic_sets, _ = get_intrinsic_sets(self.Gystar)
        return required_intrinsic_sets

    def _allowed_intrinsic_sets(self, experiments):
        allowed_intrinsic_sets = set()
        allowed_intrinsic_dict = dict()
        fixing_orders = dict()
        for experiment in experiments:
            swig = copy.deepcopy(self.graph)
            swig.fix(experiment)
            intrinsic_sets, order_dict = get_intrinsic_sets(swig)
            allowed_intrinsic_sets.update(intrinsic_sets)
            fixing_orders[frozenset(experiment)] = order_dict
            for s in intrinsic_sets:
                allowed_intrinsic_dict[frozenset(s)] = experiment
            # allowed_intrinsic_dict[frozenset(experiment)] = intrinsic_sets

        return allowed_intrinsic_sets, allowed_intrinsic_dict, fixing_orders

    def functional(self, experiments=[set()]):
        if not self.id(experiments=experiments):
            raise NotIdentifiedError

        # create and return the functional
        functional = '' if set(self.ystar) == set(self.outcomes) else '\u03A3'

        for y in self.ystar:
            if y not in self.outcomes:
                functional += y
        if len(self.ystar) > 1: functional += ' '

        # guarantee a deterministic printing order
        fixing = []
        items = []
        for item in self.required_intrinsic_sets:
            fixed = self.allowed_intrinsic_dict[item]
            fixing.append(list(fixed))
            items.append(item)

        sorted_items = sorted(items, key=dict(zip(items, fixing)).get)
        sorted_fixing = sorted(fixing)

        for i, item in enumerate(sorted_items):
            fixed = sorted_fixing[i]

            correct_order = self.fixing_orders[frozenset(fixed)][frozenset(item) - frozenset(fixed)]

            functional += '\u03A6' + ','.join(reversed(correct_order)) + ' p(V \ {0} | do({0}))'.format(",".join(fixed))

        return functional

    def id(self, experiments=[set()]):
        required_intrinsic_sets = self._required_intrinsic_sets()
        allowed_intrinsic_sets, allowed_intrinsic_dict, fixing_orders = self._allowed_intrinsic_sets(experiments)
        self.required_intrinsic_sets = required_intrinsic_sets
        self.allowed_intrinsic_sets = allowed_intrinsic_sets
        self.allowed_intrinsic_dict = allowed_intrinsic_dict
        self.fixing_orders = fixing_orders

        is_id = False

        if allowed_intrinsic_sets >= required_intrinsic_sets:
            is_id = True

        return is_id


if __name__ == '__main__':
    # ID test
    vertices = ['A', 'B', 'C', 'D', 'Y']
    di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
    bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
    G = ADMG(vertices, di_edges, bi_edges)
    one_id = OneLineID(G, ['A'], ['Y'])
    one_id.draw_swig(direction='LR').render()
    # G.draw(direction='LR').render()
    print(one_id.ystar)
    print(one_id.id())
    print(one_id.functional())
    one_id.export_intermediates()

    vertices = ["A", "D", "C", "Y"]
    di_edges = [('D', 'Y'), ('D', 'A'), ('A', 'Y'), ('C', 'A'), ('C', 'Y')]
    bi_edges = []
    G = ADMG(vertices, di_edges, bi_edges)
    one_id = OneLineID(G, ['A'], ['Y', 'A'])
    one_id.draw_swig(direction='LR').render()
    # G.draw(direction='LR').render()
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
