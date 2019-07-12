"""
Class to represent vertex of a graphical model.

The vertex class should be general enough to fit
into any kind of graphical model -- UGs, DAGs,
CGs, ADMGs, CPDAGs, CADMGs etc.
"""


class Vertex:

    def __init__(self, name, fixed=False, cardinality=2):
        """
        Constructor.

        :param name: name of the vertex.
        :param fixed: boolean specifying whether vertex is fixed or random.
        :param cardinality: integer indicating categories or string "continuous".
        """

        self.name = name
        self.parents = set()
        self.children = set()
        self.siblings = set()
        self.neighbors = set()
        self.fixed = fixed
        self.cardinality = cardinality

