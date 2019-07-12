"""
Class for Intrinsic Graphs (IGs) -- a mixed graph used to compute intrinsic sets
of an ADMG in polynomial time.
"""

import logging
from .graph import Graph
import itertools


logger = logging.getLogger(__name__)


class IG(Graph):

    def __init__(self, admg):
        """
        Constructor.

        :param admg: ADMG object to calculate intrinsic sets for.
        """

        self.admg = admg
        super().__init__()

        # the IG is initialized with vertices corresponding to
        # reachable closures of singletons (these are guaranteed to be intrinsic)
        for v in admg.vertices:
            rc = frozenset(self.admg.reachable_closure(v))
            self.add_vertex(rc)

            # add di/bi edges that fulfill subset relation
            for i in self.vertices:

                if i < rc:
                    self.add_diedge(i, rc)
                elif rc < i:
                    self.add_diedge(rc, i)

                if not(i in self.ancestors([rc]) or rc in self.ancestors([i])) and self.bidirected_connected(i, rc):
                    self.add_biedge(i, rc)

        print(self.di_edges)
        print(self.bi_edges)

    def bidirected_connected(self, s1, s2):
        """
        Check if two sets are bidirected connected in the original ADMG

        :param s1: First set corresponding to a vertex in the IG.
        :param s2: Second set corresponding to a vertex in the IG.
        :return: boolean corresponding to connectedness.
        """

        possible_endpoints = itertools.product(*[s1, s2])
        for combination in possible_endpoints:
            if self.admg.has_biedge(*combination):
                return True
        return False

    def maintain_subset_relation(self, s):
        """
        Add di edges to a newly inserted vertex s so as to maintain
        the subset relation.
        :param s: Frozenset corresponding to the new vertex.
        :return: None.
        """

        for i in self.vertices:
            if i < s:
                self.add_diedge(i, s)
            elif s < i:
                self.add_diedge(s, i)

    def add_new_biedges(self, s):
        """
        Naive O(|I(G)| choose 2) implementation. Must ensure that biedges not added to ancestors.

        :param s: Frozen set corresponding to the new vertex.
        :return: None.
        """

        ancestors_s = self.ancestors([s])

        for i in self.vertices:

            if i not in ancestors_s and self.bidirected_connected(i, s):
                self.add_biedge(i, s)

    def merge(self, s1, s2):
        """
        Merge operation on two sets s1 and s2 to give a (possibly)
        new intrinsic set s3.

        :param s1: First set corresponding to a vertex in the IG.
        :param s2: Second set corresponding to a vertex in the IG.
        :return: None.
        """

        s3c = set(s1)
        s3c.update(set(s2))
        s3 = frozenset(self.admg.reachable_closure(s3c))
        self.delete_biedge(s1, s2)

        # if the intrinsic set already exists, ignore it
        if s3 in self.vertices:
            return

        # add the vertex and add di edges
        self.add_vertex(s3)
        self.maintain_subset_relation(s3)

        # add new bi edges to the added vertex
        self.add_new_biedges(s3)

    def get_intrinsic_sets(self):
        """
        Get all intrinsic sets for given ADMG.

        TODO: How to also obtain fixing orders for each intrinsic set?

        :return:
        """

        # keep merging vertices that are connected
        # by bidirected edges to obtain new intrinsic
        # sets until there are none left
        while len(self.bi_edges) > 0:

            u, v = next(iter(self.bi_edges))
            self.merge(u, v)

        return set(self.vertices)
