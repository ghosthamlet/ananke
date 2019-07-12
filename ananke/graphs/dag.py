"""
Class for Directed Acyclic Graphs (DAGs).
"""
import logging

from .admg import ADMG
from .cg import CG

logger = logging.getLogger(__name__)


class DAG(ADMG, CG):

    def __init__(self, vertices=[], di_edges=set(), **kwargs):
        """
        Constructor.

        :param vertices: iterable of names of vertices.
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y.
        """

        super().__init__(vertices=vertices, di_edges=di_edges, **kwargs)
        logger.debug("DAG")
