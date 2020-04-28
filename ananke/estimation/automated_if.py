"""
Class for automated derivation of influence functions.
"""

import copy
from .counterfactual_mean import CausalEffect


class AutomatedIF:
    """
    IF for a single treatment and single outcome.
    """

    def __init__(self, graph, treatment, outcome):
        """
        Constructor.

        :param graph: ADMG corresponding to substantive knowledge/model.
        :param treatment: name of vertex corresponding to the intervention.
        :param outcome: name of vertex corresponding to the outcome.
        """

        # without loss of generality, focus on ancestors of Y
        self.graph = copy.deepcopy(graph)
        self.treatment = treatment
        self.outcome = outcome

        self.ace = CausalEffect(self.graph, self.treatment, self.outcome)

        if self.ace.strategy == "Not ID":
            raise RuntimeError("Query is not identified.")

        self.beta_primal_ = ""
        self.beta_dual_ = ""
        self.nonparametric_if_ = ""
        self.eff_if_ = ""

        if self.ace.strategy == "a-fixable":
            self._format_augmented_ipw()

        elif self.ace.strategy == "p-fixable":
            self._format_augmented_primal_ipw()

        else:
            raise NotImplementedError("Influence functions are available only when the treatment is " +
                                      "a-fixable or p-fixable")

    def _format_density(self, V, cond_set, intervened=False, expectation=False):
        """
        Provide appropriate formatting of a p(V | conditioning set).

        :param V: vertex of interest.
        :param cond_set: set of vertices to condition on.
        :param expectation: whether it is a conditional expectation instead of density i.e. E[ | ].
        :return: string formatted appropriately.
        """

        density = ""

        if len(cond_set) == 0:
            if expectation:
                density = "E[{}]".format(V)
            else:
                density = "p({})".format(V)
        else:
            if expectation:
                density = "E[{}|{}]".format(V, ",".join(cond_set))
            else:
                density = "p({}|{})".format(V, ",".join(cond_set))

        if intervened:
            density = density.replace(self.treatment, self.treatment + "=" + self.treatment.lower())

        return density

    def _format_augmented_ipw(self):
        """
        Format the IF for augmented IPW and efficient augmented IPW if possible.

        :return: None.
        """

        # get the Markov pillow of T
        mp_T = self.graph.markov_pillow([self.treatment], self.ace.p_order)

        # format the propensity score/IPW
        ipw_form = "I({}={}) x 1/".format(self.treatment, self.treatment.lower())
        ipw_form += self._format_density(self.treatment, mp_T)
        self.beta_primal_ += ipw_form + " x {}".format(self.outcome)

        # format the outcome regression
        self.beta_dual_ += self._format_density(self.outcome, mp_T.union({self.treatment}),
                                                expectation=True, intervened=True)

        # nonparametric IF
        self.nonparametric_if_ = (ipw_form +
                                  " x ({} - {})".format(self.outcome, self.beta_dual_) +
                                  " + " + self.beta_dual_ + " - Ψ")

        # eff IF if nonparametric saturated is just nonparametric IF
        if self.graph.nonparametric_saturated():
            self.eff_if_ = self.nonparametric_if_

        # otherwise do projections if graph is mb-shielded
        elif self.ace.is_mb_shielded:
            contributions = []
            for V in set(self.graph.vertices).difference({self.treatment}):
                mpV = self.graph.markov_pillow([V], self.ace.p_order)
                VmpV = set([V]).union(mpV)
                contributions.append(self._format_density("βprimal", VmpV, expectation=True) + " - " +
                                     self._format_density("βprimal", mpV, expectation=True))
            self.eff_if_ = ' + '.join(contributions)

        # otherwise we do not know how to compute the efficient IF
        else:
            self.eff_if_ = "Cannot compute, graph is not mb-shielded."

    def _format_augmented_primal_ipw(self):
        """
        Format the IF for augmented primal IPW and efficient augmented primal IPW if possible.

        :return:
        """

        # C := pre-treatment vars and L := post-treatment vars in district of treatment
        C = self.graph.pre([self.treatment], self.ace.p_order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))

        self.beta_primal_ = "I({}={}) x 1/[".format(self.treatment, self.treatment.lower())
        primal_terms = []
        for Li in L.difference([self.outcome]):
            mpLi = self.graph.markov_pillow([Li], self.ace.p_order)
            primal_terms.append(self._format_density(Li, mpLi))
        self.beta_primal_ += ''.join(primal_terms) + '] x Σ_{} '.format(self.treatment)
        self.beta_primal_ += ''.join(primal_terms)

        if self.outcome in L:
            mpY = self.graph.markov_pillow([self.outcome], self.ace.p_order)
            self.beta_primal_ += ' x ' + self._format_density(self.outcome, mpY, expectation=True)
        else:
            self.beta_primal_ += ' x {}'.format(self.outcome)

        # M := inverse Markov pillow of the treatment
        M = set([m for m in self.graph.vertices if self.treatment in self.graph.markov_pillow([m], self.ace.p_order)])
        M = M.difference(self.graph.district(self.treatment))
        dual_terms = []
        for Mi in M.difference([self.outcome]):
            mpMi = self.graph.markov_pillow([Mi], self.ace.p_order)
            dual_fixed_term = self._format_density(Mi, mpMi, intervened=True)
            dual_random_term = self._format_density(Mi, mpMi)
            dual_terms.append('[' + dual_fixed_term + '/' + dual_random_term + ']')
        self.beta_dual_ = ' x '.join(dual_terms)

        if self.outcome in M:
            mpY = self.graph.markov_pillow([self.outcome], self.ace.p_order)
            self.beta_dual_ += ' x ' + self._format_density(self.outcome, mpY, expectation=True, intervened=True)
        else:
            self.beta_dual_ += ' x {}'.format(self.outcome)

        # nonparametric IF
        contributions = []
        for V in self.graph.vertices:

            preV = self.graph.pre([V], self.ace.p_order)
            VpreV = set([V]).union(preV)

            if V in M:
                contributions.append(self._format_density("βprimal", VpreV, expectation=True) + " - " +
                                     self._format_density("βprimal", preV, expectation=True))
            elif V in L:
                contributions.append(self._format_density("βdual", VpreV, expectation=True) + " - " +
                                     self._format_density("βdual", preV, expectation=True))
            else:
                contributions.append(self._format_density("βprimal or βdual", VpreV, expectation=True) + " - " +
                                     self._format_density("βprimal or βdual", preV, expectation=True))
        self.nonparametric_if_ = ' + '.join(contributions)

        # eff IF if nonparametric saturated is just nonparametric IF
        if self.graph.nonparametric_saturated():
            self.eff_if_ = self.nonparametric_if_

        elif self.ace.is_mb_shielded:
            contributions = []
            for V in self.graph.vertices:

                mpV = self.graph.markov_pillow([V], self.ace.p_order)
                VmpV = set([V]).union(mpV)

                if V in M:
                    contributions.append(self._format_density("βprimal", VmpV, expectation=True) + " - " +
                                         self._format_density("βprimal", mpV, expectation=True))
                elif V in L:
                    contributions.append(self._format_density("βdual", VmpV, expectation=True) + " - " +
                                         self._format_density("βdual", mpV, expectation=True))
                else:
                    contributions.append(self._format_density("βprimal or βdual", VmpV, expectation=True) + " - " +
                                         self._format_density("βprimal or βdual", mpV, expectation=True))
            self.eff_if_ = ' + '.join(contributions)

        # otherwise we do not know how to compute the efficient IF
        else:
            self.eff_if_ = "Cannot compute, graph is not mb-shielded."
