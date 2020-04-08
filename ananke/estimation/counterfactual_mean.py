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

    def _aipw(self, data, assignment, model_binary=None, model_continuous=None):

        if self.strategy != "a-fixable":
            return RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.order)

        # Fit T | mp(T)
        formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
        model = model_binary(data, formula)
        prob_T = model.predict(data)
        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # Fit Y | T=t, mp(T)
        formula = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T) + "+ ones"
        data_assign = data.copy()
        data_assign[self.treatment] = assignment
        if set([0, 1]).issuperset(data[self.outcome].unique()):
            model = model_binary(data, formula)
        else:
            model = model_continuous(data, formula)
        Yhat_vec = model.predict(data_assign)

        return np.mean((indices / prob_T) * (Y - Yhat_vec) + Yhat_vec)

    def _eif_augmented_ipw(self, data, assignment, model_binary=None, model_continuous=None):

        if self.strategy != "a-fixable":
            return RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")
        if not self.is_mb_shielded:
            return RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.order)

        # Fit T | mp(T)
        formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
        model = model_binary(data, formula)
        prob_T = model.predict(data)
        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # Compute projections
        primal = (indices / prob_T) * Y
        data["primal"] = primal
        eif_vec = 0

        eif_vars = [v for v in self.graph.vertices]
        eif_vars.remove(self.treatment)
        # eif_vars = ['Y', 'M', 'C2', 'C1']

        for v in eif_vars:
            mpV = self.graph.markov_pillow([v], self.order)
            formula = "primal ~ " + '+'.join(mpV)
            if len(mpV) == 0:
                primal_mpV = np.mean(primal)
            else:
                model_mpV = model_continuous(data, formula)
                primal_mpV = model_mpV.predict(data)
            formula = formula + "+" + v
            model_VmpV = model_continuous(data, formula)
            primal_VmpV = model_VmpV.predict(data)
            eif_vec += primal_VmpV - primal_mpV

        eif_vec = eif_vec + np.mean(primal)

        return np.mean(eif_vec)

    def _beta_primal(self, data, assignment, model_binary=None, model_continuous=None):

        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        Y = data[self.outcome]
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        # M = post - L

        data_T1 = data.copy()
        data_T1[self.treatment] = 1
        data_T0 = data.copy()
        data_T0[self.treatment] = 0

        indices = data[self.treatment] == assignment
        prob = 1
        prob_T1 = 1
        prob_T0 = 1
        for V in L.difference([self.outcome]):
            # Fit V | mp(V)
            mp_V = self.graph.markov_pillow([V], self.order)
            formula = V + " ~ " + '+'.join(mp_V) + "+ ones"
            model = model_binary(data, formula)
            indices_V0 = data.index[data[V] == 0]
            # p(V | .)
            prob_V = model.predict(data)
            prob_V[indices_V0] = 1 - prob_V[indices_V0]
            # p(V | . ) when T=1
            prob_V_T1 = model.predict(data_T1)
            prob_V_T1[indices_V0] = 1 - prob_V_T1[indices_V0]
            # p(V | . ) when T=0
            prob_V_T0 = model.predict(data_T0)
            prob_V_T0[indices_V0] = 1 - prob_V_T0[indices_V0]

            prob *= prob_V
            prob_T1 *= prob_V_T1
            prob_T0 *= prob_V_T0

        if self.outcome in L:
            mp_Y = self.graph.markov_pillow(self.outcome, self.order)
            formula = self.outcome + " ~ " + '+'.join(mp_Y) + "+ ones"
            if set([0, 1]).issuperset(data[self.outcome].unique()):
                model = model_binary(data, formula)
            else:
                model = model_continuous(data, formula)
            Yhat_T1 = model.predict(data_T1)
            Yhat_T0 = model.predict(data_T0)
            prob_sumT = prob_T1*Yhat_T1 + prob_T0*Yhat_T0
            beta_primal = indices*(prob_sumT/prob)
        else:
            prob_sumT = prob_T1 + prob_T0
            beta_primal = indices * (prob_sumT / prob)*Y

        # TODO: fix fitting a binary vs a continuous variable
        return beta_primal

    def _primal_ipw(self, data, assignment, model_binary=None, model_continuous=None):

        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Primal IPW will not return valid estimates as treatment is not p-fixable")
        return np.mean(self._beta_primal(data, assignment, model_binary, model_continuous))

    def _beta_dual(self, data, assignment, model_binary=None, model_continuous=None):

        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        Y = data[self.outcome]
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L - set([self.outcome])
        M = set([m for m in M if self.treatment in self.graph.markov_pillow([m], self.order)])

        data_assigned = data.copy()
        data_assigned[self.treatment] = assignment

        prob = 1
        for V in M.difference([self.outcome]):
            # Fit V | mp(V)
            mp_V = self.graph.markov_pillow([V], self.order)
            formula = V + " ~ " + '+'.join(mp_V) + "+ ones"
            model = model_binary(data, formula)
            indices_V0 = data.index[data[V] == 0]
            # p(V | .)
            prob_V = model.predict(data)
            prob_V[indices_V0] = 1 - prob_V[indices_V0]
            # p(V | . ) when T=assignment
            prob_V_assigned = model.predict(data_assigned)
            prob_V_assigned[indices_V0] = 1 - prob_V_assigned[indices_V0]

            prob *= prob_V/prob_V_assigned

        if self.outcome in M:
            mp_Y = self.graph.markov_pillow(self.outcome, self.order)
            formula = self.outcome + " ~ " + '+'.join(mp_Y) + "+ ones"
            if set([0, 1]).issuperset(data[self.outcome].unique()):
                model = model_binary(data, formula)
            else:
                model = model_continuous(data, formula)
            Yhat_assigned = model.predict(data_assigned)
        else:
            Yhat_assigned = Y

        # TODO: fix fitting a binary vs a continuous variable
        return prob*Yhat_assigned

    def _dual_ipw(self, data, assignment, model_binary=None, model_continuous=None):

        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Dual IPW will not return valid estimates as treatment is not p-fixable")
        return np.mean(self._beta_dual(data, assignment, model_binary, model_continuous))

    def _augmented_primal_ipw(self, data, assigned, model_binary=None, model_continuous=None):

        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        beta_primal = self._beta_primal(data, assigned, model_binary, model_continuous)
        beta_dual = self._beta_dual(data, assigned, model_binary, model_continuous)
        data["beta_primal"] = beta_primal
        data["beta_dual"] = beta_dual

        Y = data[self.outcome]
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L

        IF = 0
        for V in post:
            pre_V = self.graph.pre([V], self.order)
            if V in M:
                formula = "beta_primal" + " ~ " + '+'.join(pre_V)
            elif V in L:
                formula = "beta_dual" + " ~ " + '+'.join(pre_V)
            if len(pre_V) != 0:
                model_preV = model_continuous(data, formula)
                pred_preV = model_preV.predict(data)
            else:
                pred_preV = 0
            formula = formula + " + " + V
            model_VpreV = model_continuous(data, formula)
            pred_VpreV = model_VpreV.predict(data)

            IF += pred_VpreV - pred_preV

        if len(C) != 0:
            formula = "beta_dual" + " ~ " + '+'.join(C)
            model_C = model_continuous(data, formula)
            IF += model_C.predict(data)
        return np.mean(IF)

    def _eif_augmented_primal_ipw(self, data, assigned, model_binary=None, model_continuous=None):

        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")
        if not self.is_mb_shielded:
            return RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        beta_primal = self._beta_primal(data, assigned, model_binary, model_continuous)
        beta_dual = self._beta_dual(data, assigned, model_binary, model_continuous)
        data["beta_primal"] = beta_primal
        data["beta_dual"] = beta_dual

        Y = data[self.outcome]
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L

        IF = 0
        for V in self.graph.vertices:
            mp_V = self.graph.markov_pillow([V], self.order)
            if V in M:
                formula = "beta_primal" + " ~ " + '+'.join(mp_V)
            else:
                formula = "beta_dual" + " ~ " + '+'.join(mp_V)

            if len(mp_V) == 0:
                pred_mpV = np.mean(beta_dual)
            else:
                model_mpV = model_continuous(data, formula)
                pred_mpV = model_mpV.predict(data)
            formula = formula + " + " + V
            model_VmpV = model_continuous(data, formula)
            pred_VmpV = model_VmpV.predict(data)
            IF += pred_VmpV - pred_mpV

        IF += np.mean(beta_dual)

        return np.mean(IF)

    def _nested_ipw(self, data, assigned, model_binary=None, model_continuous=None):

        if self.strategy == "Not ID":
            return RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm
        return 0

    def _augmented_nested_ipw(self, data, assigned, model_binary=None, model_continuous=None):

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
        print(ace_vec)