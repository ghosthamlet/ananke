"""
Class for Intrinsic Graphs (IGs) -- a mixed graph used to compute intrinsic sets
of an ADMG in polynomial time.
"""

import logging
from .graph import Graph
from .dag import DAG

logger = logging.getLogger(__name__)


class IG(Graph):

    def __init__(self, admg):
        """
        Constructor.

        :param admg: ADMG object to calculate intrinsic sets for.
        """

        self.admg = admg
        self.digraph = DAG()

        # the IG is initialized with vertices corresponding to
        # reachable closures of singletons (these are guaranteed to be intrinsic)
        vertices = [frozenset(v) for v in admg.vertices]
        retained_bi_edges = []


        #self.digraph = networkx.DiGraph()
        rc_vertices = []
        for v in vertices:
            print(frozenset(admg.reachable_closure(v)))
            reachable_closure = frozenset(admg.reachable_closure(v))


            self.digraph.add_vertex(reachable_closure)
            rc_vertices.append(reachable_closure)
        for u, v in admg.bi_edges:
            u_rc = frozenset(admg.reachable_closure(u))
            v_rc = frozenset(admg.reachable_closure(v))
            print("u", u_rc)
            print("v", v_rc)
            retained_bi_edges.append((u_rc, v_rc))

        logger.debug(retained_bi_edges)

        super().__init__(vertices=rc_vertices, bi_edges=retained_bi_edges, di_edges=set())

    def insert(self, vertex):
        for v in set(self.vertices):
            if v.issubset(vertex):
                self.digraph.add_diedge(v, vertex)
            elif v.issuperset(vertex):
                self.digraph.add_diedge(vertex, v)

        digraph_edges = self.digraph.edges()
        self.add_vertex(vertex)
        for u, v in self.di_edges.copy():
            if (u, v) not in digraph_edges:
                self.delete_diedge(u, v)

        for u, v in digraph_edges:
            self.add_diedge(u, v)

    def find_candidate_neighbors(self, v, vertex):
        for node in v:
            for i in self.admg.district(node):
                if i in vertex and v not in self.ancestors([vertex]):
                    return True
        return False

    def add_extra_biedges(self, vertex):
        """
        Naive O(|I(G)| choose 2) implementation. Must ensure that biedges not added to ancestors.
        :param vertex:
        :return:
        """
        candidate_neighbours = set()
        for v in set(self.vertices):
            if self.find_candidate_neighbors(v, vertex):
                candidate_neighbours.add(v)

        for c in candidate_neighbours:
            self.add_biedge(c, vertex)

    def merge(self, vertex1, vertex2):

        s3c = set(vertex1)
        s3c.update(set(vertex2))
        s3 = frozenset(self.admg.reachable_closure(s3c))

        self.delete_biedge(vertex1, vertex2)

        if s3 in set(self.vertices):
            return None
        else:
            self.insert(s3)

        self.add_extra_biedges(vertex=s3)

        return None

    def get_intrinsic_sets(self):
        """
        Get all intrinsic sets for given ADMG.

        TODO: How to also obtain fixing orders for each intrinsic set?

        :return:
        """
        #print(self.bi_edges)
        while len(self.bi_edges) > 0:
            #print(self.bi_edges)
            u, v = next(iter(self.bi_edges))
            self.merge(u, v)

        return set(self.vertices)
