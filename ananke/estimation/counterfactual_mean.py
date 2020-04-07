"""
Estimate the counterfactual mean E[Y(t)]
"""

import statsmodels.api as sm
from statsmodels.gam.generalized_additive_model import GLMGam

from ananke.identification import OneLineID

class CounterfactualMean:
    """
    estimation strategies for E[Y(t)]
    """
    def __init__(self, graph, treatment, outcome, order=[]):
        """
        Constructor.

        :param graph: ADMG corresponding to substantive knowledge/model
        :param treatment: name of vertex corresponding to the treatment
        :param outcome: name of vertex corresponding to the outcome
        """

        self.graph = graph
        self.treatment = treatment
        self.outcome = outcome
        self.order = order
        self.strategy = None

        # Check if the query is ID
        one_id = OneLineID(graph, [treatment], [outcome])

        # Check the fixability criteria for the treatment
        if len(self.graph.district(treatment).intersection(self.graph.descendants([treatment]))) == 1:
            self.strategy = "a-fixable"
            print("a-fixable")

        elif len(self.graph.district(treatment).intersection(self.graph.children([treatment]))) == 0:
            self.strategy = "p-fixable"
            print("p-fixable")

        elif one_id.id():
            self.strategy = "nested-fixable"
            print("nested-fixable")

        else:
            print("not ID")

    def estimate(self, data, value_T):

        if self.strategy == "a-fixable":
            mp_T = self.graph.markov_pillow([self.treatment], self.order)
            formula = self.treatment + " ~ " + '+'.join(mp_T)
            print(sm.families)
            model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial()).fit()
            proba_T = model.predict(data)
            print(proba_T)
            indices = data[self.treatment] == value_T
            print(indices)

            print(mp_T)