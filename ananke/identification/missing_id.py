"""
Class for missing ID
"""


class MissingFullID:

    def __init__(self, graph):
        """
        Constructor

        :param graph: Missing data graph to run the ID algorithm on
        """

        self.graph = graph

    def id(self):
        """
        Function to ID the full law

        :return: boolean is ID or not
        """

        for Ri in self.graph.vertices:

            # Only look at the missingness indicators in the list of vertices
            if not Ri.startswith('R_'):
                continue

            # Find children of Ri that are not in proxy set Xp
            child_Ri = set([ v for v in self.graph.children([Ri]) if v.startswith('R_') ])

            # Find Markov blanket of the set {Ri, child_Ri} and add back child_Ri to the set
            colluding_path_Ri = self.graph.markov_blanket(set.union({Ri}, child_Ri)).union(child_Ri)

            # Find Xi (the counterfactual X with the corresponding missingness indicator Ri)
            Xi = 'X_' + Ri.split('_')[1]

            # If Xi is in colluding_path_Ri, then there is a colluding path between Ri and Xi.
            # Therefore, full law is not ID.
            if Xi in colluding_path_Ri:
                return False

        return True
