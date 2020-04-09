"""
Class that provides an interface to estimation strategies for the counterfactual mean E[Y(t)]
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.gam.generalized_additive_model import GLMGam
from ananke.identification import OneLineID
import copy


class AverageCausalEffect:
    """
    Provides an interface to various estimation strategies for the ACE: E[Y(1) - Y(0)].
    """

    def __init__(self, graph, treatment, outcome, order=[]):
        """
        Constructor.

        :param graph: ADMG corresponding to substantive knowledge/model.
        :param treatment: name of vertex corresponding to the treatment.
        :param outcome: name of vertex corresponding to the outcome.
        """

        self.graph = copy.deepcopy(graph)
        self.treatment = treatment
        self.outcome = outcome
        self.order = order
        self.strategy = None
        self.is_mb_shielded = self.graph.mb_shielded()

        # a dictionary of names for available estimators
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

        # a dictionary of names for available modeling strategies
        self.models = {"glm-binary": self._fit_binary_glm,
                       "glm-continuous": self._fit_continuous_glm}

        # check if the query is ID
        one_id = OneLineID(graph, [treatment], [outcome])

        # check the fixability criteria for the treatment
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

    def _fit_binary_glm(self, data, formula):
        """
        Fit a binary GLM to data given a formula.

        :param data: pandas data frame containing the data.
        :param formula: string encoding an R-style formula e.g: Y ~ X1 + X2.
        :return: the fitted model.
        """

        return sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial()).fit()

    def _fit_continuous_glm(self, data, formula):
        """
        Fit a linear GLM to data given a formula.

        :param data: pandas data frame containing the data.
        :param formula: string encoding an R-style formula: e.g. Y ~ X1 + X2.
        :return: the fitted model.
        """

        return sm.GLM.from_formula(formula, data=data, family=sm.families.Gaussian()).fit()

    def _ipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for propensity score: e.g. glm-binary.
        :param model_continuous: this argument is ignored for IPW.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "a-fixable":
            return RuntimeError("IPW will not return valid estimates as treatment is not a-fixable")

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm

        # extract outcome from data frame and compute Markov pillow of treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.order)

        # fit T | mp(T) and compute probability of treatment for each sample
        formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
        model = model_binary(data, formula)
        prob_T = model.predict(data)
        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]

        # compute IPW estimate
        indices = data[self.treatment] == assignment
        return np.mean((indices / prob_T) * Y)

    def _gformula(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Outcome regression estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for outcome regression: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "a-fixable":
            return RuntimeError("g-formula will not return valid estimates as treatment is not a-fixable")

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # fit Y | T=t, mp(T)
        mp_T = self.graph.markov_pillow([self.treatment], self.order)
        formula = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T) + "+ ones"

        # create a dataset where T=t
        data_assign = data.copy()
        data_assign[self.treatment] = assignment

        # predict outcome appropriately depending on binary vs continuous
        if set([0, 1]).issuperset(data[self.outcome].unique()):
            model = model_binary(data, formula)
        else:
            model = model_continuous(data, formula)

        # return E[Y(t)]
        return np.mean(model.predict(data_assign))

    def _aipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Augmented IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "a-fixable":
            return RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # extract the outcome and get Markov pillow of the treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.order)

        # fit T | mp(T) and predict treatment probabilities
        formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
        model = model_binary(data, formula)
        prob_T = model.predict(data)
        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # fit Y | T=t, mp(T) and predict outcomes under assignment T=t
        formula = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T) + "+ ones"
        data_assign = data.copy()
        data_assign[self.treatment] = assignment
        if set([0, 1]).issuperset(data[self.outcome].unique()):
            model = model_binary(data, formula)
        else:
            model = model_continuous(data, formula)
        Yhat_vec = model.predict(data_assign)

        # return AIPW estimate
        return np.mean((indices / prob_T) * (Y - Yhat_vec) + Yhat_vec)

    def _eif_augmented_ipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Efficient augmented IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "a-fixable":
            return RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")
        if not self.is_mb_shielded:
            return RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # extract the outcome and get Markov pillow of treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.order)

        # fit T | mp(T) and compute treatment probabilities
        formula = self.treatment + " ~ " + '+'.join(mp_T) + "+ ones"
        model = model_binary(data, formula)
        prob_T = model.predict(data)
        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # compute the primal estimates that we use to compute projections
        primal = (indices / prob_T) * Y
        data["primal"] = primal
        eif_vec = 0

        # get the variables that are actually involved in the efficient influence function
        # TODO: prune extra variables
        eif_vars = [v for v in self.graph.vertices]
        eif_vars.remove(self.treatment)

        # iterate over variables
        for v in eif_vars:

            # get the Markov pillow of the variable
            mpV = self.graph.markov_pillow([v], self.order)

            # compute E[primal | mp(V)]
            formula = "primal ~ " + '+'.join(mpV)

            # special case if mp(V) is empty, then E[primal | mp(V)] = E[primal]
            if len(mpV) == 0:
                primal_mpV = np.mean(primal)
            else:
                model_mpV = model_continuous(data, formula)
                primal_mpV = model_mpV.predict(data)

            # compute E[primal | V, mp(V)]
            formula = formula + "+" + v
            model_VmpV = model_continuous(data, formula)
            primal_VmpV = model_VmpV.predict(data)

            # add contribution of current variable: E[primal | V, mp(V)] - E[primal | mp(V)]
            eif_vec += primal_VmpV - primal_mpV

        # re-add the primal so final result is not mean zero
        eif_vec = eif_vec + np.mean(primal)

        # return efficient AIPW estimate
        return np.mean(eif_vec)

    def _beta_primal(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Utility function to compute primal estimates for a dataset.

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: numpy array of floats corresponding to primal estimates for each sample.
        """

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # extract the outcome
        Y = data[self.outcome]

        # C := pre-treatment vars and L := post-treatment vars in district of treatment
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))

        # create copies of the data with treatment assignments T=0 and T=1
        data_T1 = data.copy()
        data_T1[self.treatment] = 1
        data_T0 = data.copy()
        data_T0[self.treatment] = 0

        indices = data[self.treatment] == assignment
        prob = 1  # stores \prod_{Li in L} p(Li | mp(Li))
        prob_T1 = 1  # stores \prod_{Li in L} p(Li | mp(Li)) at T=1
        prob_T0 = 1  # stores \prod_{Li in L} p(Li | mp(Li)) at T=0

        # iterate over vertices in L (except the outcome)
        for V in L.difference([self.outcome]):

            # fit V | mp(V)
            mp_V = self.graph.markov_pillow([V], self.order)
            formula = V + " ~ " + '+'.join(mp_V) + "+ ones"
            model = model_binary(data, formula)
            indices_V0 = data.index[data[V] == 0]

            # p(V | .)
            prob_V = model.predict(data)
            prob_V[indices_V0] = 1 - prob_V[indices_V0]

            # p(V | . ) when T=1
            prob_V_T1 = model.predict(data_T1)
            # special case for treatment which is set to be 1
            if V != self.treatment:
                prob_V_T1[indices_V0] = 1 - prob_V_T1[indices_V0]

            # p(V | . ) when T=0
            prob_V_T0 = model.predict(data_T0)
            # special case for treatment which is set to be 0
            if V != self.treatment:
                prob_V_T0[indices_V0] = 1 - prob_V_T0[indices_V0]
            else:
                prob_V_T0 = 1 - prob_V_T0

            prob *= prob_V
            prob_T1 *= prob_V_T1
            prob_T0 *= prob_V_T0

        # special case when the outcome is in L
        if self.outcome in L:

            # fit a binary/continuous model for Y | mp(Y)
            mp_Y = self.graph.markov_pillow([self.outcome], self.order)
            formula = self.outcome + " ~ " + '+'.join(mp_Y) + "+ ones"
            if set([0, 1]).issuperset(data[self.outcome].unique()):
                model = model_binary(data, formula)
            else:
                model = model_continuous(data, formula)

            # predict the outcome and adjust numerator of primal accordingly
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
        """
        Primal IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Primal IPW will not return valid estimates as treatment is not p-fixable")

        # return primal IPW estimate
        return np.mean(self._beta_primal(data, assignment, model_binary, model_continuous))

    def _beta_dual(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Utility function to compute dual estimates for a dataset.

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: numpy array of floats corresponding to dual estimates for each sample.
        """

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # extract the outcome
        Y = data[self.outcome]

        # M := inverse Markov pillow of the treatment
        M = set([m for m in self.graph.vertices if self.treatment in self.graph.markov_pillow([m], self.order)])

        # create a dataset where T=t
        data_assigned = data.copy()
        data_assigned[self.treatment] = assignment

        prob = 1  # stores \prod_{Mi in M} p(Mi | mp(Mi))|T=t / p(Mi | mp(Mi))
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

            prob *= prob_V_assigned/prob_V

        # special case for if the outcome is in M
        if self.outcome in M:
            mp_Y = self.graph.markov_pillow([self.outcome], self.order)
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
        """
        Dual IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Dual IPW will not return valid estimates as treatment is not p-fixable")

        # return primal dual IPW estimate
        return np.mean(self._beta_dual(data, assignment, model_binary, model_continuous))

    def _augmented_primal_ipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Augmented primal IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # compute the primal and dual estimates and them to the data frame
        beta_primal = self._beta_primal(data, assignment, model_binary, model_continuous)
        beta_dual = self._beta_dual(data, assignment, model_binary, model_continuous)
        data["beta_primal"] = beta_primal
        data["beta_dual"] = beta_dual

        # C := pre-treatment vars
        # L := post-treatment vars in the district of T
        # M := post treatment vars not in L (a.k.a. the rest)
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L

        IF = 0

        # iterate over all post-treatment variables
        for V in post:

            # compute all predecessors according to the topological order
            pre_V = self.graph.pre([V], self.order)

            # if the variables is in M, project using the primal otherwise use the dual
            # to fit E[beta | pre(V)]
            if V in M:
                formula = "beta_primal" + " ~ " + '+'.join(pre_V)
            elif V in L:
                formula = "beta_dual" + " ~ " + '+'.join(pre_V)

            # special logic for if there are no predecessors for the variable
            if len(pre_V) != 0:
                model_preV = model_continuous(data, formula)
                pred_preV = model_preV.predict(data)
            else:
                pred_preV = 0

            # fit E[beta | V, pre(V)]
            formula = formula + " + " + V
            model_VpreV = model_continuous(data, formula)
            pred_VpreV = model_VpreV.predict(data)

            # add contribution of current variable as E[beta | V, pre(V)] - E[beta | pre(V)]
            IF += pred_VpreV - pred_preV

        # final contribution from E[beta | C] (if C is not empty)
        if len(C) != 0:
            formula = "beta_dual" + " ~ " + '+'.join(C)
            model_C = model_continuous(data, formula)
            IF += model_C.predict(data)

        # return APIPW estimate
        return np.mean(IF)

    def _eif_augmented_primal_ipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Efficient augmented primal IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy != "p-fixable" and self.strategy != "a-fixable":
            return RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")
        if not self.is_mb_shielded:
            return RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # compute primal and dual estimates and add them to the data frame
        beta_primal = self._beta_primal(data, assignment, model_binary, model_continuous)
        beta_dual = self._beta_dual(data, assignment, model_binary, model_continuous)
        data["beta_primal"] = beta_primal
        data["beta_dual"] = beta_dual

        # C := pre-treatment vars
        # L := post-treatment vars in the district of T
        # M := post treatment vars not in L (a.k.a. the rest)
        C = self.graph.pre([self.treatment], self.order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L

        IF = 0
        # iterate over all variables
        for V in self.graph.vertices:

            # get the Markov pillow
            mp_V = self.graph.markov_pillow([V], self.order)

            # if the variables is in M, project using the primal otherwise use the dual
            # to fit E[beta | mp(V)]
            if V in M:
                formula = "beta_primal" + " ~ " + '+'.join(mp_V)
            else:
                formula = "beta_dual" + " ~ " + '+'.join(mp_V)

            # special logic for if there the Markov pillow is empty
            if len(mp_V) == 0:
                pred_mpV = np.mean(beta_dual)
            else:
                model_mpV = model_continuous(data, formula)
                pred_mpV = model_mpV.predict(data)

            # fit E[beta | V, mp(V)]
            formula = formula + " + " + V
            model_VmpV = model_continuous(data, formula)
            pred_VmpV = model_VmpV.predict(data)

            # add contribution of current variable as E[beta | V, mp(V)] - E[beta | mp(V)]
            IF += pred_VmpV - pred_mpV

        # add final contribution so that estimator is not mean-zero
        IF += np.mean(beta_dual)

        # return efficient APIPW estimate
        return np.mean(IF)

    def _nested_ipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Nested IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        # pedantic checks to make sure the method returns valid estimates
        if self.strategy == "Not ID":
            return RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm
        return 0

    def _augmented_nested_ipw(self, data, assignment, model_binary=None, model_continuous=None):
        """
        Augmented nested IPW estimator for the counterfactual mean E[Y(t)].

        :param data: pandas data frame containing the data.
        :param assignment: assignment value for treatment.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: float corresponding to computed E[Y(t)].
        """

        if self.strategy == "Not ID":
            return RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")
        return 0

    def bootstrap_ace(self, data, estimator, model_binary=None, model_continuous=None, n_bootstraps=10):
        """
        Bootstrap functionality to compute the Average Causal Effect
        :param data: pandas data frame containing the data.
        :param estimator: string indicating what estimator to use: e.g. eif-apipw.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :param n_bootstraps: number of bootstraps.
        :return: None
        """

        # add a column of ones to fit intercept terms
        data['ones'] = np.ones(len(data))

        # instantiate estimator and get point estimate of ACE
        method = self.estimators[estimator]
        point_estimate_T1 = method(data, 1, model_binary, model_continuous)
        point_estimate_T0 = method(data, 0, model_binary, model_continuous)
        ace = point_estimate_T1 - point_estimate_T0
        ace_vec = [ace]

        # iterate over bootstraps
        for iter in range(n_bootstraps):

            # resample the data with replacement
            data_sampled = data.sample(len(data), replace=True)
            data_sampled.reset_index(drop=True, inplace=True)

            # estimate ACE in resampled data
            estimate_T1 = method(data_sampled, 1, model_binary, model_continuous)
            estimate_T0 = method(data_sampled, 0, model_binary, model_continuous)
            ace_vec.append(estimate_T1 - estimate_T0)

        # Quantile calculation
        quantiles = np.quantile(ace_vec, q=[0.025, 0.975])
        print("ACE = ", ace)
        print("(2.5%, 97.5%) = ", "(", quantiles[0], ",", quantiles[1], ")")