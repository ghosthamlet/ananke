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

    def format_density(self, Vi, intervened=True):
        """
        Provide appropriate formatting of a p(Vi | mp(Vi))

        :param Vi: vertex of interest
        :return: string formatted appropriately
        """

        pillow_Vi = self.graph.markov_pillow([Vi], self.top_order_)
        density = ""

        if len(pillow_Vi) == 0:
            density =  "p({})".format(Vi)
        else:
            density = "p({}|{})".format(Vi, ",".join(pillow_Vi))

        if Vi in self.M and intervened:
            density = density.replace(self.A, self.A.lower())

        return density

    def compute_if(self):
        """
        Compute the IF.

        :return: a string corresponding to the IF.
        """

        # keep a dictionary of IF terms
        IF_terms = {}
        IF = ""

        # sum over each Mi
        for Mi in self.M:

            IF_term = ""
            print("Processing", Mi)

            # get all Li < Mi
            pre_Mi = set(self.graph.pre([Mi], self.top_order_)).intersection(self.L.union({self.A}))

            # get all Vi > Mi
            post_Mi = set(self.graph.vertices) - set(self.graph.pre([Mi], self.top_order_)) - set({Mi})

            # process the denominator under the indicator I(A=a)
            denominator = ""
            for Li in pre_Mi:
                denominator += self.format_density(Li)

            IF_term += (r"\frac{\mathbb{I}(A=a)}{%s}" % (denominator))
            IF_term += "[ "

            # add summation over A, Vi > Mi
            summation = (r"\sum_{%s") % ("A")
            if len(post_Mi) > 0:
                summation += (r",%s") % (",".join(post_Mi))
            IF_term += summation + "} "

            product = "Y"
            product_Mi = post_Mi.union(self.L)
            if len(product_Mi) > 0:
                product += r"\times "
            for Vi in product_Mi:
                product += self.format_density(Vi)

            IF_term += product
            IF_term += " - " + summation + "," + Mi + "} "
            IF_term += self.format_density(Mi) + product
            IF_term += " ]"
            print(IF_term)
            IF_terms[Mi] = IF_term
            IF += "+ " + IF_term

        # sum over each Li excluding treatment
        for Li in self.L - set({self.A}):

            IF_term = ""
            print("Processing", Li)

            # get all Mi < Li
            pre_Li = set(self.graph.pre([Li], self.top_order_)).intersection(self.M)

            # get all Vi > Li
            post_Li = set(self.graph.vertices) - set(self.graph.pre([Li], self.top_order_)) - set({Li})

            # process ratio fixes
            numerator = ""
            denominator = ""
            for Mi in pre_Li:
                numerator += self.format_density(Mi)
                denominator += self.format_density(Mi, intervened=False)

            IF_term += (r"\frac{%s}{%s}") % (numerator, denominator)
            IF_term += "[ "

            if len(post_Li) > 0:
                IF_term += (r"\sum_{%s} ") % (",".join(post_Li))

            product = "Y"
            if len(post_Li) > 0:
                product += r"\times "

            for Vi in post_Li:
                product += self.format_density(Vi)

            IF_term += product
            IF_term += " - "
            IF_term += (r"\sum_{%s} ") % (",".join(set({Li}).union(post_Li)))
            IF_term += self.format_density(Li) + product
            IF_term += " ]"
            print(IF_term)
            IF_terms[Mi] = IF_term
            IF += "+ " + IF_term

        # processing treatments and post-treatments
        print("Processing treatment and pre-treatments")
        IF_term = (r"(\sum_{%s} ") % (",".join(set(self.graph.vertices) - self.C.union({self.A})))
        IF_term += r"Y\times "

        for Mi in self.M:
            IF_term += self.format_density(Mi)

        for Li in self.L - set({self.A}):
            IF_term += self.format_density(Li)

        print(IF_term)
        IF += "+ " + IF_term

        return IF
