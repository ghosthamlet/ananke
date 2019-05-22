"""
Class for segregated graphs (SGs)
"""

import copy
from .graph import Graph


class SG(Graph):

    def __init__(self, vertices, di_edges=set(), bi_edges=set(), ud_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y
        :param ud_edges: iterable of tuples of undirected edges i.e. (X, Y) = X - Y
        """

        # initialize vertices
        Graph.__init__(self, vertices=vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges)

        assert self._segregated(), "TypeError: Graph is not segregated"
        assert self._acyclic(), "TypeError: Graph is not acyclic"

        # a mapping of vertices to their district ids
        # and a list of districts
        self._district_map = {}
        self._districts = []
        self._calculate_districts()

        # a mapping of vertices to their block ids
        # and a list of blocks
        # NOTE: in an SG, only blocks of size >= 2 are considered
        self._block_map = {}
        self._blocks = []
        self._calculate_blocks()

    #### VALID GRAPH CHECKS ####
    def _acyclic(self):
        """
        Checks if the graph is directed and partially directed cycle free

        :return: boolean indicator whether graph is acyclic
        """

        # TODO: check correctness

        for vertex in self.vertices.values():
            if self._dfs_cycle(vertex):
                return False
        return True

    def _dfs_cycle(self, vertex):
        """
        Check if there exists a cyclic path starting at a given vertex

        :param vertex: vertex object to start DFS at
        :return: boolean indicating whether there is a cycle
        """

        visit_stack = [vertex]
        prev_vertex = None
        partially_directed = False
        visited = set()

        # DFS through children and neighbors for partially directed cycle
        while visit_stack and not visited.issuperset(visit_stack):

            visited.update(visit_stack)
            v = visit_stack.pop()
            if len(v.children) != 0:
                partially_directed = True

            visit_stack.extend(s for s in v.children.union(v.neighbors) - {prev_vertex})
            prev_vertex = v

            # if we arrive at the same vertex, then we have a cycle
            if vertex in visit_stack and partially_directed:
                return True

        return False

    def _segregated(self):
        """
        Checks if graph is segregated i.e. lacks Z <-> X - Y

        :return: boolean indicator whether graph is segregated
        """

        # TODO: is this enough to check segregated property?
        for vertex in self.vertices.values():
            if len(vertex.siblings) > 0 and len(vertex.neighbors) > 0:
                return False
        return True

    #### DISTRICT CODE ####
    def _calculate_districts(self):
        """
        Update districts in the graph

        :return: None
        """

        self._district_map = {}
        district_counter = 0

        # Add all vertices to the district_map
        for vertex in self.vertices.values():
            if vertex not in self._district_map and not vertex.fixed:
                self._dfs_district(vertex, district_counter)
                district_counter += 1

        # Now process the district_map into a list of lists
        self._districts = [set() for _ in range(district_counter)]
        for vertex, district_id in self._district_map.items():
            self._districts[district_id].add(vertex)

    def _dfs_district(self, vertex, district_id):
        """
        DFS from vertex to discover its district

        :param vertex: vertex object to start the DFS at
        :param district_id: int corresponding to district ID

        :return: None
        """

        visit_stack = [vertex]
        while visit_stack:
            v = visit_stack.pop()
            self._district_map[v] = district_id
            visit_stack.extend(s for s in v.siblings if s not in self._district_map)

    def _district(self, vertex):
        """
        Returns the district of a vertex

        :param vertex: vertex object
        :return: set corresponding to district
        """

        return self._districts[self._district_map[vertex]]

    def district(self, vertex):
        """
        Returns the district of a vertex

        :param vertex: name of the vertex
        :return: set corresponding to district
        """

        district = self._districts[self._district_map[self.vertices[vertex]]]
        return {v.name for v in district}

    def districts(self):
        """
        Returns list of all districts in the graph

        :return: list of lists corresponding to districts in the graph
        """

        districts = []
        for district in self._districts:
            districts.append({v.name for v in district})
        return districts

    #### BLOCK CODE ####
    def _calculate_blocks(self):
        """
        Update blocks in the graph

        :return: None
        """

        self._block_map = {}
        block_counter = 0

        # Add all vertices to the district_map
        for vertex in self.vertices.values():
            if vertex not in self._block_map and not vertex.fixed:
                self._dfs_block(vertex, block_counter)
                block_counter += 1

        # Now process the district_map into a list of lists
        self._blocks = [set() for _ in range(block_counter)]
        for vertex, block_id in self._block_map.items():
            self._blocks[block_id].add(vertex)

    def _dfs_block(self, vertex, block_id):
        """
        DFS from vertex to discover its block

        :param vertex: vertex object to start the DFS at
        :param block_id: int corresponding to block ID

        :return: None
        """

        visit_stack = [vertex]
        while visit_stack:
            v = visit_stack.pop()
            self._block_map[v] = block_id
            visit_stack.extend(s for s in v.neighbors if s not in self._block_map)

    def _block(self, vertex):
        """
        Returns the block of a vertex

        :param vertex: vertex object
        :return: set corresponding to block
        """

        return self._blocks[self._block_map[vertex]]

    def block(self, vertex):
        """
        Returns the block of a vertex

        :param vertex: name of the vertex
        :return: set corresponding to block
        """

        block = self._block[self._block_map[self.vertices[vertex]]]
        return {v.name for v in block}

    def blocks(self):
        """
        Returns list of all blocks in the graph

        :return: list of lists corresponding to blocks in the graph
        """

        blocks = []
        for block in self._blocks:
            blocks.append({v.name for v in block})
        return blocks

    def add_biedge(self, sib1, sib2, recompute=True):
        """
        Add a bidirected edge to the graph. Overridden to recompute districts

        :param sib1: head1 of edge
        :param sib2: head2 of edge
        :param recompute: boolean indicating whether districts should be recomputed
        :return: None
        """

        super().add_biedge(sib1, sib2)
        if recompute: self._calculate_districts()

    def delete_biedge(self, sib1, sib2, recompute=True):
        """
        Delete given bidirected edge from the graph. Overridden to recompute districts

        :param sib1: head1 of edge
        :param sib2: head2 of edge
        :param recompute: boolean indicating whether districts should be recomputed
        :return: None
        """

        super().add_biedge(sib1, sib2)
        if recompute: self._calculate_districts()

    def add_udedge(self, neb1, neb2, recompute=True):
        """
        Add an undirected edge to the graph. Overridden to recompute blocks

        :param neb1: tail1 of edge
        :param neb2: tail2 of edge
        :param recompute: boolean indicating whether blocks should be recomputed
        :return: None
        """

        super().add_udedge(neb1, neb2)
        if recompute: self._calculate_blocks()

    def delete_udedge(self, neb1, neb2, recompute=True):
        """
        Delete given undirected edge from the graph. Overridden to recompute blocks

        :param neb1: tail1 of edge
        :param neb2: tail2 of edge
        :param recompute: boolean indicating whether blocks should be recomputed
        :return: None
        """

        super().delete_udedge(neb1, neb2)
        if recompute: self._calculate_blocks()

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

            # delete undirected edges
            neighbors = [n.name for n in self.vertices[v].neighbors]
            for n in neighbors:
                if n.fixed:
                    self.delete_udedge(n, v, recompute=False)

        # recompute the districts and blocks as they may have changed
        self._calculate_districts()
        self._calculate_blocks()

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

# some simple tests for the segregated graph class
#if __name__ == "__main__":
#
#    try:
#        # non-segregated graph
#        vertices = ['A', 'B', 'C']
#        bi_edges = [('A', 'B')]
#        ud_edges = [('B', 'C')]
#        G = SG(vertices, bi_edges=bi_edges, ud_edges=ud_edges)
#        print('-'*10)
#
#    except AssertionError as error:
#        print(error)
#
#    try:
#        # directed cycle graph
#        vertices = ['A', 'B', 'C']
#        bi_edges = [('A', 'B')]
#        di_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
#        G = SG(vertices, bi_edges=bi_edges, di_edges=di_edges)
#        print(G.blocks())
#        print('-' * 10)
#
#    except AssertionError as error:
#        print(error)
#
#    try:
#        # partially directed cycle graph
#        vertices = ['A', 'B', 'C']
#        bi_edges = []
#        ud_edges = [('A', 'B'), ('B', 'C')]
#        di_edges = [('C', 'A')]
#        G = SG(vertices, bi_edges=bi_edges, di_edges=di_edges, ud_edges=ud_edges)
#        print('-' * 10)
#
#    except AssertionError as error:
#        print(error)
#
#    # undirected cycle graph this is valid
#    vertices = ['A', 'B', 'C']
#    bi_edges = []
#    ud_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
#    G = SG(vertices, bi_edges=bi_edges, ud_edges=ud_edges)
#    print(G.blocks())
#    print('-' * 10)
#
#
#
#    # simple tests
#    vertices = ['A', 'B', 'C', 'D', 'Y']
#    di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
#    bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
#    G = SG(vertices, di_edges, bi_edges)
#    print(G.districts())
#    print(G.district('A'))
#
#    vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
    di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
                ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
    bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
    G = SG(vertices, di_edges, bi_edges)
    print(G.districts())
    print(G.district('X2'))
    print(G.ancestors('A2'))
    print(G.ancestors('A1'))
    print(G.ancestors(['A1', 'A2']))
    print(G.descendants(['A1', 'A2']))
    #G.draw().render()
