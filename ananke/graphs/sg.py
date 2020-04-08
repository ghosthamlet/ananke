"""
Class for segregated graphs (SGs).
"""
import copy
import logging

from .graph import Graph

logger = logging.getLogger(__name__)


class SG(Graph):

    def __init__(self, vertices=[], di_edges=set(), bi_edges=set(), ud_edges=set(), **kwargs):
        """
        Constructor

        :param vertices: iterable of names of vertices.
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y.
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y.
        :param ud_edges: iterable of tuples of undirected edges i.e. (X, Y) = X - Y.
        """

        # initialize vertices in SG
        super().__init__(vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges, **kwargs)
        logger.debug("SG")
        if not self._segregated():
            raise TypeError("Graph is not segregated")
        if not self._acyclic():
            raise TypeError("TypeError: Graph is not acyclic")

        # a mapping of vertices to their district ids
        # and a list of districts
        self._district_map = {}
        self._districts = []

        # a mapping of vertices to their block ids
        # and a list of blocks
        # NOTE: in an SG, only blocks of size >= 2 are considered
        self._block_map = {}
        self._blocks = []

    #### VALID GRAPH CHECKS ####
    def _acyclic(self):
        """
        Checks if the graph is directed and partially directed cycle free.

        :return: boolean indicator whether graph is acyclic.
        """

        # TODO: check correctness

        for vertex in self.vertices.values():
            if self._dfs_cycle(vertex):
                return False
        return True

    def _dfs_cycle(self, vertex):
        """
        Check if there exists a cyclic path starting at a given vertex.

        :param vertex: vertex object to start DFS at.
        :return: boolean indicating whether there is a cycle.
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
        Checks if graph is segregated i.e. lacks Z <-> X - Y.

        :return: boolean indicator whether graph is segregated.
        """

        # TODO: is this enough to check segregated property?
        for vertex in self.vertices.values():
            if len(vertex.siblings) > 0 and len(vertex.neighbors) > 0:
                return False
        return True

    #### DISTRICT CODE ####
    @property
    def districts(self):
        """
        Returns list of all districts in the graph.

        :return: list of sets corresponding to districts in the graph.
        """
        return self._calculate_districts()

    def _calculate_districts(self):
        """
        Update districts in the graph.

        :return: List of districts.
        """

        self._district_map = {}
        district_counter = 0

        # Add all vertices to the district_map
        for vertex in self.vertices:
            if vertex not in self._district_map and not self.vertices[vertex].fixed:
                self._dfs_district(self.vertices[vertex], district_counter)
                district_counter += 1

        # Now process the district_map into a list of lists
        districts = [set() for _ in range(district_counter)]
        for vertex, district_id in self._district_map.items():
            districts[district_id].add(vertex)
        self._districts = districts
        return districts

    def _dfs_district(self, vertex, district_id):
        """
        DFS from vertex to discover its district.

        :param vertex: vertex object to start the DFS at.
        :param district_id: int corresponding to district ID

        :return: None
        """

        visit_stack = [vertex]
        while visit_stack:
            v = visit_stack.pop()
            self._district_map[v.name] = district_id
            visit_stack.extend(s for s in v.siblings if s.name not in self._district_map)

    def district(self, vertex):
        """
        Returns the district of a vertex.

        :param vertex: name of the vertex.
        :return: set corresponding to district.
        """

        if not self._districts:
            self.districts
        return self._districts[self._district_map[vertex]]

    #### BLOCK CODE ####
    @property
    def blocks(self):
        """
        Returns list of all blocks in the graph.

        :return: list of sets corresponding to blocks in the graph.
        """
        return self._calculate_blocks()

    def _calculate_blocks(self):
        """
        Update blocks in the graph.

        :return: None.
        """

        self._block_map = {}
        block_counter = 0

        # Add all vertices to the district_map
        for vertex in self.vertices:
            if vertex not in self._block_map and not self.vertices[vertex].fixed:
                self._dfs_block(self.vertices[vertex], block_counter)
                block_counter += 1

        # Now process the district_map into a list of lists
        self._blocks = [set() for _ in range(block_counter)]
        for vertex, block_id in self._block_map.items():
            self._blocks[block_id].add(vertex)

        return self._blocks

    def _dfs_block(self, vertex, block_id):
        """
        DFS from vertex to discover its block.

        :param vertex: vertex object to start the DFS at.
        :param block_id: int corresponding to block ID.

        :return: None
        """

        visit_stack = [vertex]
        while visit_stack:
            v = visit_stack.pop()
            self._block_map[v.name] = block_id
            visit_stack.extend(n for n in v.neighbors if n.name not in self._block_map)

    def block(self, vertex):
        """
        Returns the block of a vertex.

        :param vertex: name of the vertex.
        :return: set corresponding to block.
        """
        if not self._blocks:
            self.blocks
        block = self._blocks[self._block_map[vertex]]
        return block

    def add_biedge(self, sib1, sib2, recompute=True):
        """
        Add a bidirected edge to the graph. Overridden to recompute districts.

        :param sib1: endpoint 1 of edge.
        :param sib2: endpoint 2 of edge.
        :param recompute: boolean indicating whether districts should be recomputed.
        :return: None.
        """

        super().add_biedge(sib1, sib2)
        if recompute: self._districts = self._calculate_districts()

    def delete_biedge(self, sib1, sib2, recompute=True):
        """
        Delete given bidirected edge from the graph. Overridden to recompute districts.

        :param sib1: endpoint 1 of edge.
        :param sib2: endpoint 2 of edge.
        :param recompute: boolean indicating whether districts should be recomputed.
        :return: None.
        """

        super().delete_biedge(sib1, sib2)
        if recompute: self._districts = self._calculate_districts()

    def add_udedge(self, neb1, neb2, recompute=True):
        """
        Add an undirected edge to the graph. Overridden to recompute blocks

        :param neb1: endpoint 1 of edge.
        :param neb2: endpoint 2 of edge.
        :param recompute: boolean indicating whether blocks should be recomputed.
        :return: None.
        """

        super().add_udedge(neb1, neb2)
        if recompute: self._blocks = self._calculate_blocks()

    def delete_udedge(self, neb1, neb2, recompute=True):
        """
        Delete given undirected edge from the graph. Overridden to recompute blocks.

        :param neb1: endpoint 1 of edge.
        :param neb2: endpoint 2 of edge.
        :param recompute: boolean indicating whether blocks should be recomputed.
        :return: None.
        """

        super().delete_udedge(neb1, neb2)
        if recompute: self._blocks = self._calculate_blocks()

    def fix(self, vertices):
        """
        Perform the graphical operation of fixing on a set of vertices.

        :param vertices: iterable of vertices to be fixed.
        :return: None.
        """

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
        self._districts = self._calculate_districts()
        self._blocks = self._calculate_blocks()

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

                # check if any nodes are reachable via -> AND <->
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
