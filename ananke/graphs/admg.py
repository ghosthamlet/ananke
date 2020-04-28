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
    """
    Class for creating and manipulating (conditional) acyclic directed mixed graphs (ADMGs/CADMGs).
    """

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

    def markov_blanket(self, vertices):
        """
        Get the Markov blanket of a set of vertices.

        :param vertices: iterable of vertex names.
        :return: set corresponding to Markov blanket.
        """

        blanket = set()
        for v in vertices:
            blanket = blanket.union(self.district(v))
        blanket = blanket.union(self.parents(blanket))
        return blanket - set(vertices)

    @property
    def fixed(self):
        """
        Returns all fixed nodes in the graph.

        :return:
        """
        fixed_vertices = []
        for v in self.vertices:
            if self.vertices[v].fixed:
                fixed_vertices.append(v)

        return fixed_vertices

    def is_subgraph(self, other):
        """
        Check that this graph is a subgraph of other, meaning it has  a subset of edges and nodes of the other.

        :param other: an object of the ADMG class.
        :return: boolean indicating whether the statement is True or not.
        """
        if set(self.vertices).issubset(set(other.vertices)) and \
                set(self.di_edges).issubset(set(other.di_edges)) and \
                set(self.bi_edges).issubset(set(other.bi_edges)):
            return True
        return False

    def is_ancestral_subgraph(self, other):
        """
        Check that this graph is an ancestral subgraph of the other.
        An ancestral subgraph over variables S and intervention b G(S(b)) of a larger graph G(V(b)) is defined as a
        subgraph, such that ancestors of each node s in S with respect to the graph G(V(b_i)) are contained in S.

        :param other: an object of the ADMG class.
        :return: boolean indicating whether the statement is True or not.
        """
        if not self.is_subgraph(other):
            return False

        for v in self.vertices:
            self_parents = set([item.name for item in self.vertices[v].parents])
            other_parents = set([item.name for item in other.vertices[v].parents])
            if self_parents != other_parents:
                return False

        return True

    def fix(self, vertices):
        # TODO: there should only be one fixing operation implemented in SG
        """
        Perform the graphical operation of fixing on a set of vertices.

        :param vertices: name(s) of vertices to be fixed.
        :return: None.
        """

        # if isinstance(vertices, str):
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

        # initialize set of vertices that must still be fixed
        remaining_vertices = set(self.vertices) - set(vertices) - set(
            v for v in self.vertices if self.vertices[v].fixed)
        fixing_order = []  # keep track of the valid fixing order
        fixed = True  # flag to track that a vertex was successfully fixed in a given pass
        G = copy.deepcopy(self)

        # keep iterating over remaining vertices until there are no more or we failed to fix
        while remaining_vertices and fixed:

            fixed = False

            # check if any remaining vertex van be fixed
            for v in remaining_vertices:

                # fixability check
                if len(G.descendants([v]).intersection(G.district(v))) == 1:
                    G.fix([v])
                    remaining_vertices.remove(v)
                    fixing_order.append(v)
                    fixed = True  # flag that we succeeded
                    break  # stop the current pass over vertices

        # compute final reachable closure based on vertices successfully fixed
        reachable_closure = set(G.vertices) - set(v for v in G.vertices if G.vertices[v].fixed)

        # return the reachable closure, the valid order, and the resulting CADMG
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

        # create an intrinsic set graph and obtain the intrinsic sets + valid fixing orders leading to them
        ig = IG(copy.deepcopy(self))
        intrinsic_sets = ig.get_intrinsic_sets()
        fixing_orders = ig.iset_fixing_order_map

        return intrinsic_sets, fixing_orders

    def maximal_arid_projection(self):
        """
        Get the maximal arid projection that encodes the same conditional independences and
        Vermas as the original ADMG. This operation is described in Acyclic
        Linear SEMs obey the Nested Markov property by Shpitser et al 2018.

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

    def mb_shielded(self):
        """
        Check if the ADMG is a Markov blanket shielded ADMG. That is, check if
        two vertices are non-adjacent only when they are absent from each others'
        Markov blankets.

        :return: boolean indicating if it is mb-shielded or not.
        """

        # iterate over all pairs of vertices
        for Vi, Vj in itertools.combinations(self.vertices, 2):
            # check if the pair is not adjacent
            if not (Vi in self.siblings([Vj]) or (Vi, Vj) in self.di_edges or (Vj, Vi) in self.di_edges):
                # if one is in the Markov blanket of the other, then it is not mb-shielded
                if Vi in self.markov_blanket([Vj]) or Vj in self.markov_blanket([Vi]):
                    return False
        return True

    def nonparametric_saturated(self):
        """
        Check if the nested Markov model implied by the ADMG is nonparametric saturated.
        The following is an implementation of Algorithm 1 in Semiparametric Inference for
        Causal Effects in Graphical Models with Hidden Variables (Bhattacharya, Nabi & Shpitser 2020)
        which was shown to be sound and complete for this task.

        :return: boolean indicating if it is nonparametric saturated or not.
        """

        # iterate over all pairs of vertices
        for Vi, Vj in itertools.combinations(self.vertices, 2):

            # check if there is no dense inducing path between Vi and Vj
            # and return not NPS if either of the checks fail
            if not (Vi in self.parents(self.reachable_closure([Vj])[0]) or
                    Vj in self.parents(self.reachable_closure([Vi])[0]) or
                    Vi in self.reachable_closure([Vi, Vj])[2].district(Vj)):
                return False
        return True
