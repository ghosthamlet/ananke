import logging

import networkx

from .admg import ADMG

logger = logging.getLogger(__name__)


class IG(ADMG):

    def __init__(self, admg):
        self.admg = admg
        vertices = [frozenset(v) for v in admg.vertices]
        retained_bi_edges = []

        for u, v in admg.bi_edges:
            if not admg.descendants(u).intersection(v) and not admg.descendants(v).intersection(u):
                retained_bi_edges.append((frozenset(u), frozenset(v)))

        self.digraph = networkx.DiGraph()
        for v in vertices:
            self.digraph.add_node(v)

        super().__init__(vertices=vertices, bi_edges=retained_bi_edges, di_edges=set())

    def insert(self, vertex):
        for v in set(self.vertices):
            if v.issubset(vertex):
                self.digraph.add_edge(v, vertex)
            elif v.issuperset(vertex):
                self.digraph.add_edge(vertex, v)

        self.digraph = networkx.algorithms.dag.transitive_reduction(self.digraph)

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
        Naive O(|I(G)| choose 2) implementation
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
        s3 = frozenset(self.admg.get_reachable_closure(s3c))

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
        while len(self.bi_edges) > 0:
            u, v = next(iter(self.bi_edges))
            self.merge(u, v)

        return set(self.vertices)
