"""
Class for acyclic directed mixed graphs (ADMGs) and conditional ADMGs (CADMGs)
"""

import copy
from .sg import SG


class ADMG(SG):

    def __init__(self, vertices, di_edges=set(), bi_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y
        """

        # initialize vertices
        super().__init__(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)

    def fix(self, vertices):
        """
        Perform the graphical operation of fixing on a set of vertices

        :param vertices: Name(s) of vertices to be fixed
        :return: None
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
        self._calculate_districts()

    def fixable(self, vertices):
        """
        Check if there exists a valid fixing order and return such
        an order in the form of a list, else returns an empty list

        :param vertices:
        :return: A boolean indicating whether the set was fixable and a valid fixing order as a stack
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
        Get all m-connecting paths between x and y after conditioning on Z using BFS

        :param x: name of vertex x
        :param y: name of vertex y
        :param Z: name of set of vertices Z being conditioned on
        :return: list of m-connecting paths
        """

        queue = [(self.vertices[x], [self.vertices[x]], [])]
        y = self.vertices[y]
        Z = [self.vertices[z] for z in Z]
        ancestors_z = self._ancestors(Z)

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
        the specified vertices and edges between them)

        :param vertices: set containing names of vertices in the subgraph
        :return: a new Graph object corresponding to the subgraph
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


if __name__ == "__main__":

    # simple tests
    vertices = ['A', 'B', 'C', 'D', 'Y']
    di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
    bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
    G = ADMG(vertices, di_edges, bi_edges)
    print(G.districts())
    print(G.district('A'))

    vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
    di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
                ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
    bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
    G = ADMG(vertices, di_edges, bi_edges)
    print(G.districts())
    print(G.district('X2'))
    #G.draw().render()
    paths = list(G.m_connecting_paths('X1', 'Y2'))
    print('-------------' * 5)
    for path in paths:
        print(path)
    paths = list(G.m_connecting_paths('X1', 'Y2', set(['U', 'A1'])))
    print('-------------'*5)
    for path in paths:
        print(path)

    paths = list(G.m_connecting_paths('X1', 'Y2', set(['U', 'A1', 'X2'])))
    print('-------------' * 5)
    for path in paths:
        print(path)