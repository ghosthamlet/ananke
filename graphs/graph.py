"""
Base class for all graphs

TODO: Add error checking
"""


from graphs.vertex import Vertex
from graphviz import Digraph


class Graph:

    def __init__(self, vertices, di_edges=set(), bi_edges=set(), ud_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y
        :param ud_edges: iterable of tuples of undirected edges i.e. (X, Y) = X - Y
        """

        # initialize vertices
        self.vertices = {v: Vertex(v) for v in vertices}

        # initialize edges
        self.di_edges = set()
        self.bi_edges = set()
        self.ud_edges = set()

        # read in directed edges
        for parent, child in di_edges:
            self.add_diedge(parent, child)

        # read in bidirected edges
        for sib1, sib2 in bi_edges:
            self.add_biedge(sib1, sib2)

        # read in undirected edges
        for neb1, neb2 in ud_edges:
            self.add_udedge(neb1, neb2)

    def add_vertex(self, name):
        """
        Add a vertex to the graph

        :param name: name of vertex
        :return: None
        """

        self.vertices[name] = Vertex(name)

    def add_diedge(self, parent, child):
        """
        Add a directed edge to the graph

        :param parent: tail of edge
        :param child: head of edge
        :return: None
        """

        self.di_edges.add((parent, child))
        self.vertices[parent].children.add(self.vertices[child])
        self.vertices[child].parents.add(self.vertices[parent])

    def delete_diedge(self, parent, child):
        """
        Deleted given directed edge from the graph

        :param parent: tail of edge
        :param child: head of edge
        :return: None
        """

        self.di_edges.remove((parent, child))
        self.vertices[parent].children.remove(self.vertices[child])
        self.vertices[child].parents.remove(self.vertices[parent])

    def add_biedge(self, sib1, sib2):
        """
        Add a bidirected edge to the graph

        :param sib1: head1 of edge
        :param sib2: head2 of edge
        :return: None
        """

        self.bi_edges.add((sib1, sib2))
        self.vertices[sib1].siblings.add(self.vertices[sib2])
        self.vertices[sib2].siblings.add(self.vertices[sib1])

    def delete_biedge(self, sib1, sib2):
        """
        Delete given bidirected edge from the graph

        :param sib1: head1 of edge
        :param sib2: head2 of edge
        :return: None
        """

        try:
            self.bi_edges.remove((sib1, sib2))
        except KeyError:
            self.bi_edges.remove((sib2, sib1))

        self.vertices[sib1].siblings.remove(self.vertices[sib2])
        self.vertices[sib2].siblings.remove(self.vertices[sib1])

    def add_udedge(self, neb1, neb2):
        """
        Add an undirected edge to the graph

        :param neb1: tail1 of edge
        :param neb2: tail2 of edge
        :return: None
        """

        self.ud_edges.add((neb1, neb2))
        self.vertices[neb1].neighbors.add(self.vertices[neb2])
        self.vertices[neb2].neighbors.add(self.vertices[neb1])

    def delete_udedge(self, neb1, neb2):
        """
        Delete given undirected edge from the graph

        :param neb1: tail1 of edge
        :param neb2: tail2 of edge
        :return: None
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
        Get parents of a vertex or set of vertices

        :param vertices: vertex name or iterable of vertex names
        :return: set of parents
        """

        if isinstance(vertices, str):
            return {p.name for p in self.vertices[vertices].parents}
        else:
            parents = set()
            for v in vertices:
                for p in self.vertices[v].parents:
                    parents.add(p.name)
            return parents

    def children(self, vertices):
        """
        Get children of a vertex or set of vertices

        :param vertices: vertex name or iterable of vertex names
        :return: set of children
        """

        if isinstance(vertices, str):
            return {c.name for c in self.vertices[vertices].children}
        else:
            children = set()
            for v in vertices:
                for c in self.vertices[v].children:
                    children.add(c.name)
            return children

    def neighbors(self, vertices):
        """
        Get neighbors of a vertex or set of vertices

        :param vertices: vertex name or iterable of vertex names
        :return: set of neighbors
        """

        if isinstance(vertices, str):
            return {n.name for n in self.vertices[vertices].neighbors}
        else:
            neighbors = set()
            for v in vertices:
                for n in self.vertices[v].neighbors:
                    neighbors.add(n.name)
            return neighbors

    def siblings(self, vertices):
        """
        Get siblings of a vertex or set of vertices

        :param vertices: vertex name or iterable of vertex names
        :return: set of neighbors
        """

        if isinstance(vertices, str):
            return {s.name for s in self.vertices[vertices].siblings}
        else:
            siblings = set()
            for v in vertices:
                for s in self.vertices[v].siblings:
                    siblings.add(s.name)
            return siblings

    def _ancestors(self, vertices):
        """
        Get the ancestors of a vertex or set of vertices

        :param vertices: single vertex objects or iterable of vertex objects to find ancestors for
        :return: set of ancestors
        """

        ancestors = set()

        if isinstance(vertices, Vertex):
            visit_stack = [vertices]
        else:
            visit_stack = list(vertices)
        while visit_stack:
            v = visit_stack.pop()
            ancestors.add(v)
            visit_stack.extend(p for p in v.parents if p not in ancestors)

        return ancestors

    def ancestors(self, vertices):
        """
        Get ancestors of a vertex or set of vertices

        :param vertices: vertex name or iterable of vertex names
        :return: set of ancestors
        """

        if isinstance(vertices, str):
            return {a.name for a in self._ancestors(self.vertices[vertices])}
        else:
            return {a.name for a in self._ancestors([self.vertices[v] for v in vertices])}

    def _descendants(self, vertices):
        """
        Get the descendants of a vertex or set of vertices

        :param vertices: single vertex objects or iterable of vertex objects to find descendants for
        :return: set of descendants
        """

        descendants = set()

        if isinstance(vertices, Vertex):
            visit_stack = [vertices]
        else:
            visit_stack = list(vertices)
        while visit_stack:
            v = visit_stack.pop()
            descendants.add(v)
            visit_stack.extend(c for c in v.children if c not in descendants)

        return descendants

    def descendants(self, vertices):
        """
        Get descendants of a vertex or set of vertices

        :param vertices: vertex name or iterable of vertex names
        :return: set of descendants
        """

        if isinstance(vertices, str):
            return {d.name for d in self._descendants(self.vertices[vertices])}
        else:
            return {d.name for d in self._descendants([self.vertices[v] for v in vertices])}

    def subgraph(self, vertices):
        """
        Return a subgraph on the given vertices (i.e. a graph containing only
        the specified vertices and edges between them)

        :param vertices: set containing names of vertices in the subgraph
        :return: a new Graph object corresponding to the subgraph
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

    def draw(self):
        """
        Visualize the graph

        :return : dot language representation of the graph
        """

        dot = Digraph()
        for v in self.vertices.values():
            dot.node(v.name, shape='square' if v.fixed else 'circle')

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
