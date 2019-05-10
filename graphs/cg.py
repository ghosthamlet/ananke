"""
Class for Lauritzen-Wermuth-Frydenberg chain graphs (LWF-CGs/CGs)
"""


from graphs.sg import SG


class CG(SG):

    def __init__(self, vertices, di_edges=set(), ud_edges=set()):
        """
        Constructor

        :param vertices: iterable of names of vertices
        :param di_edges: iterable of tuples of directed edges i.e. (X, Y) = X -> Y
        :param ud_edges: iterable of tuples of undirected edges i.e. (X, Y) = X - Y
        """

        # initialize vertices
        SG.__init__(self, vertices=vertices, di_edges=di_edges, ud_edges=ud_edges)
