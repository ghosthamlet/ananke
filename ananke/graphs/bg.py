"""
Class for Bidirected Graphs (BGs)
"""


from .admg import ADMG


class BG(ADMG):

    def __init__(self, vertices, bi_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param bi_edges: iterable of tuples of bidirected edges i.e. (X, Y) = X <-> Y
        """

        # initialize vertices
        ADMG.__init__(self, vertices=vertices, bi_edges=bi_edges)
