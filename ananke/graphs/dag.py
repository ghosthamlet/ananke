"""
Class for Directed Acyclic Graphs (DAGs)
"""


from .admg import ADMG
from .cg import CG


class DAG(ADMG, CG):

    def __init__(self, vertices, di_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y
        """

        # initialize vertices
        ADMG.__init__(self, vertices=vertices, di_edges=di_edges)
        CG.__init__(self, vertices=vertices, di_edges=di_edges)
