"""
Base class for all graphs.

TODO: Add error checking
"""

import copy
from .vertex import Vertex


class Graph:

    def __init__(self, vertices=[], di_edges=set(), bi_edges=set(), ud_edges=set(), **kwargs):
        """
        Constructor.

        :param vertices: iterable of names of vertices.
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y.
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y.
        :param ud_edges: iterable of tuples of undirected edges i.e. (X, Y) = X - Y.
        """
        assert not kwargs, "Unrecognised kwargs: {}".format(kwargs)

        # initialize vertices
        self.vertices = {v: Vertex(v) for v in vertices}

        # initialize edges
        self.di_edges = set()
        self.bi_edges = set()
        self.ud_edges = set()

        # read in directed edges
        # explicit reference to Graph so that overridden functions
        # aren't called that make calls to recompute districts etc.
        for parent, child in di_edges:
            Graph.add_diedge(self, parent, child)

        # read in bidirected edges
        for sib1, sib2 in bi_edges:
            Graph.add_biedge(self, sib1, sib2)

        # read in undirected edges
        for neb1, neb2 in ud_edges:
            Graph.add_udedge(self, neb1, neb2)

    def add_vertex(self, name):
        """
        Add a vertex to the graph.

        :param name: name of vertex.
        :return: None.
        """

        self.vertices[name] = Vertex(name)

    def add_diedge(self, parent, child):
        """
        Add a directed edge to the graph.

        :param parent: tail of edge.
        :param child: head of edge.
        :return: None.
        """

        self.di_edges.add((parent, child))
        self.vertices[parent].children.add(self.vertices[child])
        self.vertices[child].parents.add(self.vertices[parent])

    def delete_diedge(self, parent, child):
        """
        Deleted given directed edge from the graph.

        :param parent: tail of edge.
        :param child: head of edge.
        :return: None.
        """

        self.di_edges.remove((parent, child))
        self.vertices[parent].children.remove(self.vertices[child])
        self.vertices[child].parents.remove(self.vertices[parent])

    def add_biedge(self, sib1, sib2):
        """
        Add a bidirected edge to the graph.

        :param sib1: endpoint 1 of edge.
        :param sib2: endpoint 2 of edge.
        :return: None.
        """

        self.bi_edges.add((sib1, sib2))
        self.vertices[sib1].siblings.add(self.vertices[sib2])
        self.vertices[sib2].siblings.add(self.vertices[sib1])

    def delete_biedge(self, sib1, sib2):
        """
        Delete given bidirected edge from the graph.

        :param sib1: endpoint 1 of edge.
        :param sib2: endpoint 2 of edge.
        :return: None.
        """

        try:
            self.bi_edges.remove((sib1, sib2))
        except KeyError:
            self.bi_edges.remove((sib2, sib1))

        self.vertices[sib1].siblings.remove(self.vertices[sib2])
        self.vertices[sib2].siblings.remove(self.vertices[sib1])

    def has_biedge(self, sib1, sib2):
        """
        Check existence of a bidirected edge.

        :param sib1: endpoint 1 of edge.
        :param sib2: endpoint 2 of edge.
        :return: boolean result of existence.
        """

        if (sib1, sib2) in self.bi_edges or (sib2, sib1) in self.bi_edges:
            return True
        return False

    def add_udedge(self, neb1, neb2):
        """
        Add an undirected edge to the graph.

        :param neb1: endpoint 1 of edge.
        :param neb2: endpoint 2 of edge.
        :return: None.
        """

        self.ud_edges.add((neb1, neb2))
        self.vertices[neb1].neighbors.add(self.vertices[neb2])
        self.vertices[neb2].neighbors.add(self.vertices[neb1])

    def delete_udedge(self, neb1, neb2):
        """
        Delete given undirected edge from the graph.

        :param neb1: endpoint 1 of edge.
        :param neb2: endpoint 2 of edge.
        :return: None.
        """

        try:
            self.ud_edges.remove((neb1, neb2))
        except KeyError:
            self.ud_edges.remove((neb2, neb1))

        self.vertices[neb1].neighbors.remove(self.vertices[neb2])
        self.vertices[neb2].neighbors.remove(self.vertices[neb1])

    #### GENEALOGICAL HELPERS ####
    def parents(self, vertices):
        """
        Get parents of a vertex or set of vertices.

        :param vertices: iterable of vertex names.
        :return: set of parents.
        """

        parents = set()
        for v in vertices:
            for p in self.vertices[v].parents:
                parents.add(p.name)
        return parents

    def children(self, vertices):
        """
        Get children of a vertex or set of vertices.

        :param vertices: iterable of vertex names.
        :return: set of children.
        """

        children = set()
        for v in vertices:
            for c in self.vertices[v].children:
                children.add(c.name)
        return children

    def neighbors(self, vertices):
        """
        Get neighbors of a vertex or set of vertices.

        :param vertices: iterable of vertex names.
        :return: set of neighbors.
        """

        neighbors = set()
        for v in vertices:
            for n in self.vertices[v].neighbors:
                neighbors.add(n.name)
        return neighbors

    def siblings(self, vertices):
        """
        Get siblings of a vertex or set of vertices.

        :param vertices: vertex name or iterable of vertex names.
        :return: set of neighbors.
        """

        siblings = set()
        for v in vertices:
            for s in self.vertices[v].siblings:
                siblings.add(s.name)
        return siblings

    def ancestors(self, vertices):
        """
        Get the ancestors of a vertex or set of vertices.

        :param vertices: single vertex name or iterable of vertex names to find ancestors for.
        :return: set of ancestors.
        """

        ancestors = set()

        visit_stack = list([self.vertices[v] for v in vertices])

        while visit_stack:
            v = visit_stack.pop()
            ancestors.add(v.name)
            visit_stack.extend(p for p in v.parents if p not in ancestors)

        return ancestors

    def descendants(self, vertices):
        """
        Get the descendants of a vertex or set of vertices.

        :param vertices: single vertex name or iterable of vertex names to find descendants for.
        :return: set of descendants.
        """

        descendants = set()

        visit_stack = list([self.vertices[v] for v in vertices])

        while visit_stack:
            v = visit_stack.pop()
            descendants.add(v.name)
            visit_stack.extend(c for c in v.children if c not in descendants)

        return descendants

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
        ud_edges = [e for e in self.ud_edges if e[0] in vertices and e[1] in vertices]
        subgraph = Graph(vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges)

        # set vertices that were fixed in the original graph to be fixed in the subgraph as well
        for v in vertices:
            if self.vertices[v].fixed:
                subgraph.vertices[v].fixed = True

        return subgraph

    def _bfs_directed_paths(self, source, sink):
        """
        Use BFS to find directed paths from a source vertex to sink vertices.

        :param source: name of a single source.
        :param sink: iterable of possibly multiple sinks.
        :return: list of directed paths.
        """

        sink_vertices = [self.vertices[v] for v in sink]
        queue = [(self.vertices[source], [])]

        while queue:

            (current_vertex, edge_path) = queue.pop(0)
            for c in current_vertex.children:
                if c in sink_vertices:
                    yield edge_path + [(current_vertex.name, c.name)]
                else:
                    queue.append((c, edge_path + [(current_vertex.name, c.name)]))

    def directed_paths(self, source, sink):
        """
        Get all directed paths between sets of source vertices and sink vertices.

        :param source: a set of vertices that serve as the source.
        :param sink: a set of vertices that serve as the sink.
        :return: list of directed paths.
        """

        # we use BFS for finding all paths
        directed_paths = []
        for u in source:
            directed_paths += self._bfs_directed_paths(u, sink)
        return directed_paths

    def topological_sort(self):
        """
        Perform a topological sort from roots (parentless nodes)
        to leaves, as ordered by directed edges on the graph.

        :return: list corresponding to a valid topological order.
        """

        # create a copy of the graph
        G = copy.deepcopy(self)

        # initialize roots -- vertices that have no incoming edges
        roots = [v for v in self.vertices if len(G.parents([v])) == 0]
        top_order = []

        # iterate until we have explored all nodes
        while roots:

            # pick a current root
            r = roots.pop()
            top_order.append(r)

            # iterate over all its children
            children = G.children([r])
            for c in children:

                # delete the edge
                G.delete_diedge(r, c)

                # if r was the only remaining parent, this is now a root
                if len(G.parents([c])) == 0:
                    roots.append(c)

        # return the order
        return top_order

    def pre(self, vertices, top_order):
        """
        Find all nodes prior to the given set of vertices under a topological order.

        :param vertices: iterable of vertex names.
        :param top_order: a valid topological order.
        :return: list corresponding to the order up until the given vertices.
        """

        # find all elements that are previous in the topological order
        # by iterating over the order until we encounter one of the vertices
        pre = []
        for v in top_order:
            if v in vertices:
                break
            pre.append(v)

        return pre

    # def post(self, vertices, top_order):
    #     """
    #     Find all nodes that succeed the given set of vertices under a topological order.
    #
    #     :param vertices: iterable of vertex names.
    #     :param top_order: a valid topological order.
    #     :return: list corresponding to the order up until the given vertices.
    #     """
    #
    #     # find all elements that are previous in the topological order
    #     # by iterating over the order until we encounter one of the vertices
    #     post = []
    #     for v in top_order:
    #         if v in vertices:
    #             break
    #         pre.append(v)
    #
    #     return post

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
            if parent.lower() == child:
                dot.edge(parent, child, style='invis')
            else:
                dot.edge(parent, child, color='blue')
        for sib1, sib2 in self.bi_edges:
            dot.edge(sib1, sib2, dir='both', color='red')
        for neb1, neb2 in self.ud_edges:
            dot.edge(neb1, neb2, dir='none', color='brown')

        return dot
