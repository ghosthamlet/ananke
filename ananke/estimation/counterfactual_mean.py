"""
Estimate the counterfactual mean E[Y(t)]
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.gam.generalized_additive_model import GLMGam
from itertools import combinations

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

    def mb_shielded(self):
        for v1, v2 in combinations(self.graph.vertices, 2):
            if not (v1 in self.graph.siblings([v2]) or (v1, v2) in self.graph.di_edges or (v2, v1) in self.graph.di_edges):
                if v1 in self.graph.markov_blanket([v2]) or v2 in self.graph.markov_blanket([v1]):
                    return False
        return True

    def estimate(self, data, assignment):

        if self.strategy == "a-fixable":
            data['ones'] = np.ones(len(data))
            T = data[self.treatment]
            Y = data[self.outcome]

            mp_T = self.graph.markov_pillow([self.treatment], self.order)

            # Fit T | mp(T)
            formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
            model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial()).fit()
            prob_T = model.predict(data)
            indices_T0 = data.index[data[self.treatment] == 0]
            prob_T[indices_T0] = 1 - prob_T[indices_T0]

            # Fit Y | T=t, mp(T)
            formula = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T) + "+ ones"

            if set([0, 1]).issuperset(data[self.outcome].unique()):
                family = sm.families.Binomial()
            else:
                family = sm.families.Gaussian()
            model = sm.GLM.from_formula(formula, data=data, family=family).fit()

            data_assign = data.copy()
            data_assign[self.treatment] = assignment
            Yhat = model.predict(data_assign)

            # IPW
            indices = data[self.treatment] == assignment
            ipw_mean = np.mean( (indices/prob_T)*Y )

            # g-formula
            gformula_mean = np.mean(Yhat)

            # gAIPW
            gaipw_mean = np.mean( (indices/prob_T)*(Y-Yhat) + Yhat )

            # efficient IF
            eff_mean = 0
            if self.mb_shielded():
                primal = (indices/prob_T)*Y
                data["primal"] = primal

                # eif_vars = [v for v in self.graph.vertices]
                # eif_vars.remove(self.treatment)
                eif_vars = ['Y', 'M', 'C2', 'C1']
                family = sm.families.Gaussian()
                for v in eif_vars:
                    mpV = self.graph.markov_pillow([v], self.order)
                    # print("Processing", v, self.graph.markov_pillow([v], self.order))
                    formula = "primal ~ " + '+'.join(mpV) # + "+ ones"
                    if len(mpV) == 0:
                        primal_mpV = np.mean(primal)
                    else:
                        model_mpV = sm.GLM.from_formula(formula, data=data, family=family).fit()
                        primal_mpV = model_mpV.predict(data)
                    formula = formula + "+" + v
                    print(formula)
                    model_VmpV = sm.GLM.from_formula(formula, data=data, family=family).fit()
                    primal_VmpV = model_VmpV.predict(data)
                    eff_mean += primal_VmpV - primal_mpV
                # var_eff = np.var(eff_mean + np.mean(primal))
                eff_mean = eff_mean + np.mean(primal)
                # print(var_eff)

            estimates = {"ipw":ipw_mean, "g-formula":gformula_mean, "g-aipw":gaipw_mean, "efficient-if":eff_mean}
            return estimates