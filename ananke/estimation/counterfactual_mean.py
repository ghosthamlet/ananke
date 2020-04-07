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
        self.is_mb_shielded = self.mb_shielded()
        self.estimators = {"ipw": self._ipw,
                           "gformula": self._gformula,
                           "aipw": self._aipw,
                           "eif-aipw": self._eif_augmented_ipw,
                           "p-ipw": self._primal_ipw,
                           "d-ipw": self._dual_ipw,
                           "apipw": self._augmented_primal_ipw,
                           "eif-apipw": self._eif_augmented_primal_ipw,
                           "n-ipw": self._nested_ipw,
                           "anipw": self._augmented_nested_ipw}
        self.models = {"glm-binary": self._fit_binary_glm,
                       "glm-continuous": self._fit_continuous_glm}

        # Check if the query is ID
        one_id = OneLineID(graph, [treatment], [outcome])

        # Check the fixability criteria for the treatment
        if len(self.graph.district(treatment).intersection(self.graph.descendants([treatment]))) == 1:
            self.strategy = "a-fixable"
            if self.is_mb_shielded:
                print("Treatment is a-fixable and graph is mb-shielded. Suggested estimator is efficient gAIPW \n" +
                      "Available estimators:\n" +
                      "IPW (ipw)\n" +
                      "Outcome regression (gformula)\n" +
                      "Generalized AIPW (aipw)\n" +
                      "Efficient generalized AIPW (eif-aipw)")
            else:
                print("Treatment is a-fixable. Suggested estimator is gAIPW \n" +
                      "Available estimators:\n" +
                      "IPW (ipw)\n" +
                      "Outcome regression (gformula)\n" +
                      "Generalized AIPW (aipw)")

        elif len(self.graph.district(treatment).intersection(self.graph.children([treatment]))) == 0:
            self.strategy = "p-fixable"
            if self.is_mb_shielded:
                print("Treatment is p-fixable and graph is mb-shielded. Suggested estimator is efficient APIPW \n" +
                      "Available estimators:\n" +
                      "Primal IPW (p-ipw)\n" +
                      "Dual IPW (d-ipw)\n" +
                      "APIPW (apipw)\n" +
                      "Efficient APIPW (eif-apipw)")
            else:
                print("Treatment is p-fixable. Suggested estimator is APIPW \n" +
                      "Available estimators:\n" +
                      "Primal IPW (p-ipw)\n" +
                      "Dual IPW (d-ipw)\n" +
                      "APIPW (apipw)")

        elif one_id.id():
            self.strategy = "nested-fixable"
            print("Effect is identified. Suggested estimator is Augmented NIPW \n" +
                  "Available estimators:\n" +
                  "Nested IPW (n-ipw)\n" +
                  "Augmented NIPW (anipw)")

        else:
            self.strategy = "Not ID"
            print("Query is not identified!")

    def mb_shielded(self):
        """
        Check if the input graph is a Markov blanket shielded ADMG.

        :return: boolean indicating if it is mb-shielded or not.
        """

        for v1, v2 in combinations(self.graph.vertices, 2):
            if not (v1 in self.graph.siblings([v2]) or (v1, v2) in self.graph.di_edges or (v2, v1) in self.graph.di_edges):
                if v1 in self.graph.markov_blanket([v2]) or v2 in self.graph.markov_blanket([v1]):
                    return False
        return True

    def _estimate_a_fixability(self, data, assignment):

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
        ipw_vec = (indices/prob_T)*Y

        # g-formula
        gformula_vec = Yhat

        # gAIPW
        gaipw_vec = (indices/prob_T)*(Y-Yhat) + Yhat

        # efficient IF
        if self.mb_shielded():
            primal = (indices/prob_T)*Y
            data["primal"] = primal
            eif_vec = 0

            eif_vars = [v for v in self.graph.vertices]
            eif_vars.remove(self.treatment)
            # eif_vars = ['Y', 'M', 'C2', 'C1']

            family = sm.families.Gaussian()
            for v in eif_vars:
                mpV = self.graph.markov_pillow([v], self.order)
                formula = "primal ~ " + '+'.join(mpV)
                if len(mpV) == 0:
                    primal_mpV = np.mean(primal)
                else:
                    model_mpV = sm.GLM.from_formula(formula, data=data, family=family).fit()
                    primal_mpV = model_mpV.predict(data)
                formula = formula + "+" + v
                model_VmpV = sm.GLM.from_formula(formula, data=data, family=family).fit()
                primal_VmpV = model_VmpV.predict(data)
                eif_vec += primal_VmpV - primal_mpV

            eif_vec = eif_vec + np.mean(primal)

        results_vec = {"ipw":ipw_vec, "g-formula":gformula_vec, "g-aipw":gaipw_vec, "efficient-if":eif_vec}
        return results_vec

    def _fit_binary_glm(self, data, formula):
        model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial()).fit()
        return model

    def _fit_continuous_glm(self, data, formula):
        model = sm.GLM.from_formula(formula, data=data, family=sm.families.Gaussian()).fit()
        return model

    def _ipw(self, data, assignment, model_binary=None, model_continuous=None):

        if self.strategy != "a-fixable":
            return RuntimeError("IPW will not return valid estimates as treatment is not a-fixable")

        if not model_binary:
            model_binary = self._fit_binary_glm

        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.order)

        # Fit T | mp(T)
        formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
        model = model_binary(data, formula)
        prob_T = model.predict(data)
        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]

        indices = data[self.treatment] == assignment
        return np.mean((indices / prob_T) * Y)

    def _gformula(self, data, assignment, model_binary=None, model_continuous=None):

        if self.strategy != "a-fixable":
            return RuntimeError("g-formula will not return valid estimates as treatment is not a-fixable")

        if not model_binary:
            model_binary = self._fit_binary_glm

        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # Fit Y | T=t, mp(T)
        mp_T = self.graph.markov_pillow([self.treatment], self.order)
        formula = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T) + "+ ones"
        data_assign = data.copy()
        data_assign[self.treatment] = assignment
        if set([0, 1]).issuperset(data[self.outcome].unique()):
            model = model_binary(data, formula)
        else:
            model = model_continuous(data, formula)
        Yhat_vec = model.predict(data_assign)
        return np.mean(Yhat_vec)

    def _aipw(self, data):

        if self.strategy != "a-fixable":
            return RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")

        return 0

    def _eif_augmented_ipw(self, data):

        if self.strategy != "a-fixable":
            return RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")
        if not self.is_mb_shielded:
            return RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")
        return 0

    def _primal_ipw(self, data):

        if self.strategy != "p-fixable" or self.strategy != "a-fixable":
            return RuntimeError("Primal IPW will not return valid estimates as treatment is not p-fixable")
        return 0

    def _dual_ipw(self, data):

        if self.strategy != "p-fixable" or self.strategy != "a-fixable":
            return RuntimeError("Dual IPW will not return valid estimates as treatment is not p-fixable")
        return 0

    def _augmented_primal_ipw(self, data):

        if self.strategy != "p-fixable" or self.strategy != "a-fixable":
            return RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")
        return 0

    def _eif_augmented_primal_ipw(self, data):

        if self.strategy != "p-fixable" or self.strategy != "a-fixable":
            return RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")
        if not self.is_mb_shielded:
            return RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")
        return 0

    def _nested_ipw(self, data):

        if self.strategy == "Not ID":
            return RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")
        return 0

    def _augmented_nested_ipw(self, data):

        if self.strategy == "Not ID":
            return RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")
        return 0

    def bootstrap_ace(self, data, estimator, model_binary=None, model_continuous=None, n_bootstraps=10):
        data['ones'] = np.ones(len(data))

        method = self.estimators[estimator]
        point_estimate_T1 = method(data, 1, model_binary, model_continuous)
        point_estimate_T0 = method(data, 0, model_binary, model_continuous)
        ace = point_estimate_T1 - point_estimate_T0
        ace_vec = [ace]
        for iter in range(n_bootstraps):
            data_sampled = data.sample(len(data), replace=True)
            data_sampled.reset_index(drop=True, inplace=True)
            estimate_T1 = method(data_sampled, 1, model_binary, model_continuous)
            estimate_T0 = method(data_sampled, 0, model_binary, model_continuous)
            ace_vec.append(estimate_T1 - estimate_T0)

        # Quantile calculation
        quantiles = np.quantile(ace_vec, q=[0.025, 0.975])
        print("ACE = ", ace)
        print("(2.5%, 97.5%) = ", "(", quantiles[0], ",", quantiles[1], ")")
