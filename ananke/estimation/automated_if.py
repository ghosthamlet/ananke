"""
Class for automated derivation of influence functions.
"""

class AutomatedIF:
    """
    IF for a single treatment and single outcome.
    """

    def __init__(self, graph, intervention, outcome):
        """
        Constructor.

        :param graph: ADMG corresponding to substantive knowledge/model
        :param intervention: name of vertex corresponding to the intervention
        :param outcome: name of vertex corresponding to the outcome
        """

        # without loss of generality, focus on ancestors of Y
        self.graph = graph.subgraph(graph.ancestors(outcome))
        self.A = intervention
        self.Y = outcome

        # TODO: more preprocessing

        # check if it is identified by childless in district criterion
        if len(self.graph.district(intervention).intersection(self.graph.children([intervention]))) != 0:
            raise NotImplementedError("Can currently automate simple IFs when A is childless in its district")

        # get a topological order and define our sets
        self.top_order_ = self._topological_order()
        self.C = set(self.graph.pre([self.A], self.top_order_)) # pre-treatment
        self.L = self.graph.district(self.A) - self.C
        self.M = set(self.graph.vertices) - self.C - self.L # principal mediators
        # if L is just A that means dis(A) intersect de(A) is just A and so we can sum over A
        if self.L == {self.A}: self.L = set()

        self.if_ = self.compute_if()

    def _topological_order(self):
        """
        Get a topological order in which the treatment appears as late as possible.

        :return: list corresponding to the topological order.
        """

        # first get a valid order
        top_order = self.graph.topological_sort()

        # post process to shift the treatment as late as possible
        shifted = True
        # parents_A = self.graph.parents([self.A])
        children_A = self.graph.children([self.A])
        i = top_order.index(self.A)
        while shifted:

            # if the next element is not a child of A, we can do the shift
            # if self.A not in self.graph.parents([top_order[i+1]]):
            if top_order[i+1] not in children_A:

                tmp = top_order[i+1]
                top_order[i+1] = top_order[i]
                top_order[i] = tmp

            # otherwise we're done
            else:
                shifted = False

        return top_order

    def compute_if(self):
        """
        Compute the IF.

        :return: a string corresponding to the IF.
        """

        IF = ""
        f = "" # the part of the identifying functional corresponding to district of A
        if len(self.L) > 0:
            f = "\sum_{} ".format(self.A)
            for Li in self.L:
                f += "p({}|{}) ".format(Li, ",".join(self.graph.markov_pillow([Li], self.top_order_)))
        print(f)

        for Mi in self.M:
            pre_Mi = set(self.graph.pre([Mi], self.top_order_)).intersection(self.L.union({self.A}))
            post_Mi = (set(self.graph.vertices) - set(self.graph.pre([Mi], self.top_order_)) -
                       set({Mi})).intersection(self.M)
            print("Processing", Mi, pre_Mi)
            denominator = ""
            for Li in pre_Mi:
                denominator += "p({}|{})".format(Li, ",".join(self.graph.markov_pillow([Li], self.top_order_)))
            IF += (r"\frac{\mathbb{I}(A=a)}{%s}" % (denominator))
            IF += "[ "
            product = ""
            for Pi in post_Mi:
                product += "p({}|{})".format(Pi, ",".join(self.graph.markov_pillow([Pi], self.top_order_)))
            if len(post_Mi) > 0:
                IF += "[ \sum_{{}}".format(",".join(post_Mi))
                IF += product

            IF += "{} - ".format(f)
            IF += ("\sum_{%s}" % ",".join(post_Mi.union({Mi})))
            IF += product
            IF += "p({}|{})".format(Mi, ",".join(self.graph.markov_pillow([Mi], self.top_order_)))
            IF += f + "]"
            print(IF)

        for Li in self.L:
            pre_Li = set(self.graph.pre([Li], self.top_order_)).intersection(self.M.union({self.A}))
            post_Mi = (set(self.graph.vertices) - set(self.graph.pre([Mi], self.top_order_)) -
                       set({Mi})).intersection(self.M)

        return IF
