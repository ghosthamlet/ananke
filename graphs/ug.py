"""
Class for Undirected Graphs (UGs)
"""


from graphs.cg import CG


class UG(CG):

    def __init__(self, vertices, ud_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param ud_edges: iterable of tuples of undirected edges i.e. (X, Y) = X - Y
        """

        # initialize vertices
        CG.__init__(self, vertices=vertices, ud_edges=ud_edges)

