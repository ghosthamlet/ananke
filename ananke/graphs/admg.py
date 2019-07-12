"""
Class for acyclic directed mixed graphs (ADMGs) and conditional ADMGs (CADMGs).
"""
import copy
import logging

from ananke.utils import powerset
from .sg import SG

logger = logging.getLogger(__name__)


class ADMG(SG):

    def __init__(self, vertices=[], di_edges=set(), bi_edges=set(), **kwargs):
        """
        Constructor.

        :param vertices: iterable of names of vertices.
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y.
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y.
        """

        # initialize vertices in ADMG
        super().__init__(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges, **kwargs)
        logger.debug("ADMG")

    def fix(self, vertices):
        """
        Perform the graphical operation of fixing on a set of vertices.

        :param vertices: name(s) of vertices to be fixed.
        :return: None.
        """

        if isinstance(vertices, str):
            vertices = [vertices]

        for v in vertices:

            self.vertices[v].fixed = True

            # delete incoming directed edges
            parents = [p.name for p in self.vertices[v].parents]
            for p in parents:
                self.delete_diedge(p, v)

            # delete bidirected edges
            siblings = [s.name for s in self.vertices[v].siblings]
            for s in siblings:
                self.delete_biedge(s, v, recompute=False)

        # recompute the districts as they may have changed
        self._districts = self._calculate_districts()

    def reachable_closure(self, vertices):
        """
        Obtain reachable closure for a set of vertices.

        :param vertices: set of vertices to get reachable closure for.
        :return: set corresponding to the reachable closure.
        """
        remaining_vertices = set(self.vertices) - set(vertices)
        fixing_order = []
        fixed = True
        G = copy.deepcopy(self)

        while remaining_vertices and fixed:
            fixed = False

            for v in remaining_vertices:
                if len(G.descendants(v).intersection(G.district(v))) == 1:
                    G.fix([v])
                    remaining_vertices.remove(v)
                    fixing_order.append(v)
                    fixed = True
                    break

        reachable_closure = (set(G.vertices) - set(fixing_order))

        return reachable_closure

    def fixable(self, vertices):
        """
        Check if there exists a valid fixing order and return such
        an order in the form of a list, else returns an empty list.

        :param vertices: set of vertices to check fixability for.
        :return: a boolean indicating whether the set was fixable and a valid fixing order as a stack.
        """

        # if it's just a single vertex we're checking it's easy
        if isinstance(vertices, str):
            if len(self.descendants(vertices).intersection(self.district(vertices))) == 1:
                return True, [vertices]
            return False, []

        remaining_vertices = set(vertices)
        fixing_order = []
        fixed = True  # flag to check if we fixed a variable on each pass
        G = copy.deepcopy(self)

        # while we have more vertices to fix, and were able to perform a fix
        while remaining_vertices and fixed:

            fixed = False

            for v in remaining_vertices:

                # Check if any nodes are reachable via -> AND <->
                # by looking at intersection of district and descendants
                if len(G.descendants(v).intersection(G.district(v))) == 1:
                    G.fix(v)
                    remaining_vertices.remove(v)
                    fixing_order.append(v)
                    fixed = True
                    break

            if not fixed:
                return False, fixing_order

        return True, fixing_order

    def m_connecting_paths(self, x, y, Z=set()):
        """
        Get all m-connecting paths between x and y after conditioning on Z using BFS.

        :param x: name of vertex x.
        :param y: name of vertex y.
        :param Z: name of set of vertices Z being conditioned on.
        :return: list of m-connecting paths.
        """

        queue = [(self.vertices[x], [self.vertices[x]], [])]
        y = self.vertices[y]
        Z = [self.vertices[z] for z in Z]
        ancestors_z = list([self.vertices[a] for a in self.ancestors(Z)])

        while queue:

            (vertex, vertex_path, edge_path) = queue.pop(0)

            for v in vertex.children - set(vertex_path):

                if vertex in Z:
                    continue

                if v == y:
                    yield edge_path + [(vertex.name, v.name, '->')]
                else:
                    queue.append((v, vertex_path + [v], edge_path + [(vertex.name, v.name, '->')]))

            # check colliders before doing parents/siblings
            # if it is a collider v must be in ancestors_z
            if edge_path and (edge_path[-1][-1] == '->' or edge_path[-1][-1] == '<->'):
                if vertex not in ancestors_z:
                    continue

            for v in vertex.parents - set(vertex_path):

                if v == y:
                    yield edge_path + [(vertex.name, v.name, '<-')]
                else:
                    queue.append((v, vertex_path + [v], edge_path + [(vertex.name, v.name, '<-')]))

            for v in vertex.siblings - set(vertex_path):

                if v == y:
                    yield edge_path + [(vertex.name, v.name, '<->')]
                else:
                    queue.append((v, vertex_path + [v], edge_path + [(vertex.name, v.name, '<->')]))

    def subgraph(self, vertices):
        """
        Return a subgraph on the given vertices (i.e. a graph containing only
        the specified vertices and edges between them).

        :param vertices: set containing names of vertices in the subgraph.
        :return: a new Graph object corresponding to the subgraph.
        """

        # keep only edges between vertices of the subgraph
        di_edges = [e for e in self.di_edges if e[0] in vertices and e[1] in vertices]
        bi_edges = [e for e in self.bi_edges if e[0] in vertices and e[1] in vertices]
        subgraph = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)

        # set vertices that were fixed in the original graph to be fixed in the subgraph as well
        for v in vertices:
            if self.vertices[v].fixed:
                subgraph.vertices[v].fixed = True

        return subgraph

    def get_intrinsic_sets(self):
        """
        Computes intrinsic sets (and returns the fixing order for each intrinsic set).


        :return:
        """
        intrinsic_sets, fixing_orders = get_intrinsic_sets(self)

        return intrinsic_sets, fixing_orders


def get_intrinsic_sets(graph):
    """
    Computes all intrinsic sets given an ADMG. Allows for fixed variables.

    :param graph: ADMG
    :return:
    """
    intrinsic = set()
    vertices = set(graph.vertices)
    order_dict = dict()
    fixed_vertices = set()
    for v in graph.vertices:
        if graph.vertices[v].fixed:
            fixed_vertices.add(v)

    for var in vertices:
        fixable, order = graph.fixable(vertices - set(var) - fixed_vertices)

        if fixable:
            intrinsic_set = frozenset([var])
            intrinsic.add(intrinsic_set)
            order_dict[intrinsic_set] = order
    for district in graph.districts:
        # There is possibly a more efficient way of doing this
        for pset in powerset(district, 2):
            fixable, order = graph.fixable(vertices - set(pset) - fixed_vertices)
            if fixable:
                intrinsic_set = frozenset(list(pset))
                intrinsic.add(intrinsic_set)
                order_dict[intrinsic_set] = order

    return intrinsic, order_dict

