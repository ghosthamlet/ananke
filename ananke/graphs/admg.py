"""
Class for acyclic directed mixed graphs (ADMGs) and conditional ADMGs (CADMGs).
"""
import copy
import logging
import itertools

from ananke.utils import powerset
from .sg import SG
from .ig import IG

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

    def markov_pillow(self, vertices, top_order):
        """
        Get the Markov pillow of a set of vertices. That is,
        the Markov blanket of the vertices given a valid topological order
        on the graph.

        :param vertices: iterable of vertex names.
        :param top_order: a valid topological order.
        :return: set corresponding to Markov pillow.
        """

        # get the subgraph corresponding to the vertices and nodes prior to them
        pre = self.pre(vertices, top_order)
        Gsub = self.subgraph(pre + list(vertices))

        # Markov pillow is the Markov blanket (dis(v) union pa(dis(v)) setminus v)
        # in this subgraph
        pillow = set()
        for v in vertices:
            pillow = pillow.union(Gsub.district(v))
        pillow = pillow.union(Gsub.parents(pillow))
        return pillow - set(vertices)

    def fix(self, vertices):
        # TODO: there should only be one fixing operation implemented in SG
        """
        Perform the graphical operation of fixing on a set of vertices.

        :param vertices: name(s) of vertices to be fixed.
        :return: None.
        """

        #if isinstance(vertices, str):
        #    vertices = [vertices]

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
        :return: set corresponding to the reachable closure, the fixing order for vertices
                 outside of the closure, and the CADMG corresponding to the closure.
        """
        remaining_vertices = set(self.vertices) - set(vertices) - set(v for v in self.vertices if self.vertices[v].fixed)
        fixing_order = []
        fixed = True
        G = copy.deepcopy(self)

        while remaining_vertices and fixed:
            fixed = False

            for v in remaining_vertices:
                if len(G.descendants([v]).intersection(G.district(v))) == 1:
                    G.fix([v])
                    remaining_vertices.remove(v)
                    fixing_order.append(v)
                    fixed = True
                    break

        reachable_closure = set(G.vertices) - set(v for v in G.vertices if G.vertices[v].fixed)

        return reachable_closure, fixing_order, G

    def fixable(self, vertices):
        """
        Check if there exists a valid fixing order and return such
        an order in the form of a list, else returns an empty list.

        :param vertices: set of vertices to check fixability for.
        :return: a boolean indicating whether the set was fixable and a valid fixing order as a stack.
        """

        # keep track of vertices still left to fix
        # and initialize a fixing order
        G = copy.deepcopy(self)
        remaining_vertices = set(vertices)
        fixing_order = []
        fixed = True  # flag to check if we fixed a variable on each pass

        # while we have more vertices to fix, and were able to perform a fix
        while remaining_vertices and fixed:

            fixed = False

            for v in remaining_vertices:

                # Check if any nodes are reachable via -> AND <->
                # by looking at intersection of district and descendants
                if len(G.descendants([v]).intersection(G.district(v))) == 1:
                    G.fix([v])
                    remaining_vertices.remove(v)
                    fixing_order.append(v)
                    fixed = True
                    break

            # if unsuccessful, return failure and
            # fixing order up until point of failure
            if not fixed:
                return False, fixing_order

        # if fixing vertices was successful, return success
        # and the fixing order
        return True, fixing_order

    # def _m_connecting_paths(self, x, y, Z=set()):
    #     """
    #     Get all m-connecting paths between x and y after conditioning on Z using BFS.
    #
    #     :param x: name of vertex x.
    #     :param y: name of vertex y.
    #     :param Z: name of set of vertices Z being conditioned on.
    #     :return: list of m-connecting paths.
    #     """
    #
    #     queue = [(self.vertices[x], [self.vertices[x]], [])]
    #     y = self.vertices[y]
    #     ancestors_z = list([self.vertices[a] for a in self.ancestors(Z)])
    #
    #     while queue:
    #
    #         (vertex, vertex_path, edge_path) = queue.pop(0)
    #
    #         for v in vertex.children - set(vertex_path):
    #
    #             if vertex in Z:
    #                 continue
    #
    #             if v == y:
    #                 yield edge_path + [(vertex.name, v.name, '->')]
    #             else:
    #                 queue.append((v, vertex_path + [v], edge_path + [(vertex.name, v.name, '->')]))
    #
    #         # check colliders before doing parents/siblings
    #         # if it is a collider v must be in ancestors_z
    #         if edge_path and (edge_path[-1][-1] == '->' or edge_path[-1][-1] == '<->'):
    #             if vertex not in ancestors_z:
    #                 continue
    #
    #         for v in vertex.parents - set(vertex_path):
    #
    #             if v == y:
    #                 yield edge_path + [(vertex.name, v.name, '<-')]
    #             else:
    #                 queue.append((v, vertex_path + [v], edge_path + [(vertex.name, v.name, '<-')]))
    #
    #         for v in vertex.siblings - set(vertex_path):
    #
    #             if v == y:
    #                 yield edge_path + [(vertex.name, v.name, '<->')]
    #             else:
    #                 queue.append((v, vertex_path + [v], edge_path + [(vertex.name, v.name, '<->')]))
    #
    # def _m_separation(self, x, y, Z=set()):
    #     """
    #     Test if x and y are m-separated after conditioning on Z.
    #
    #     :param x: name of vertex x.
    #     :param y: name of vertex y.
    #     :param Z: name of set of vertices Z being conditioned on.
    #     :return: boolean True or False.
    #     """
    #
    #     # if there are no m-connecting paths return True
    #     if len(list(self._m_connecting_paths(x, y, Z))) == 0:
    #         return True
    #     return False

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


        :return: list of intrinsic sets and fixing orders used to reach each one
        """
        ig = IG(copy.deepcopy(self))
        intrinsic_sets = ig.get_intrinsic_sets()
        fixing_orders = ig.iset_fixing_order_map

        return intrinsic_sets, fixing_orders

    def maximal_arid_projection(self):
        """
        Get the maximal arid projection that encodes the same conditional independences and
        Vermas as the original ADMG. This operation is described in Acyclic
        Linear SEMs obey the Nested Markov property.

        :return: An ADMG corresponding to the maximal arid projection.
        """

        vertices, di_edges, bi_edges = self.vertices, [], []

        # keep a cached dictionary of reachable closures and ancestors
        # for efficiency purposes
        reachable_closures = {}
        ancestors = {v: self.ancestors([v]) for v in vertices}

        # iterate through all vertex pairs
        for a, b in itertools.combinations(vertices, 2):

            # decide which reachable closure needs to be computed
            # and compute it if one vertex is an ancestor of another
            u, v, rc = None, None, None
            if a in ancestors[b]:
                u, v = a, b
            elif b in ancestors[a]:
                u, v = b, a

            # check parent condition and add directed edge if u is a parent of the reachable closure
            added_edge = False
            if u:
                if v not in reachable_closures:
                    reachable_closures[v] = self.reachable_closure([v])[0]
                rc = reachable_closures[v]
                if u in self.parents(rc):
                    di_edges.append((u, v))
                    added_edge = True

            # if neither are ancestors of each other we need to compute
            # the reachable closure of set {a, b} and check if it is
            # bidirected connected
            if not added_edge:
                rc, _, cadmg = self.reachable_closure([a, b])
                for district in cadmg.districts:
                    if rc <= district:
                        bi_edges.append((a, b))

        return ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
