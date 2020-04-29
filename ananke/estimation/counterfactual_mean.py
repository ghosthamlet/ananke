"""
Class that provides an interface to estimation strategies for the counterfactual mean E[Y(t)]
"""

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from ananke.identification import OneLineID
import copy


class CausalEffect:
    """
    Provides an interface to various estimation strategies for the ACE: E[Y(1) - Y(0)].
    """

    def __init__(self, graph, treatment, outcome):
        """
        Constructor.

        :param graph: ADMG corresponding to substantive knowledge/model.
        :param treatment: name of vertex corresponding to the treatment.
        :param outcome: name of vertex corresponding to the outcome.
        """

        self.graph = copy.deepcopy(graph)
        self.treatment = treatment
        self.outcome = outcome
        self.strategy = None
        self.is_mb_shielded = self.graph.mb_shielded()

        # a dictionary of names for available estimators
        self.estimators = {"ipw": self._ipw,
                           "gformula": self._gformula,
                           "aipw": self._aipw,
                           "eff-aipw": self._eff_augmented_ipw,
                           "p-ipw": self._primal_ipw,
                           "d-ipw": self._dual_ipw,
                           "apipw": self._augmented_primal_ipw,
                           "eff-apipw": self._eff_augmented_primal_ipw,
                           "n-ipw": self._nested_ipw,
                           "anipw": self._augmented_nested_ipw}

        # a dictionary of names for available modeling strategies
        self.models = {"glm-binary": self._fit_binary_glm,
                       "glm-continuous": self._fit_continuous_glm}

        # check if the query is ID
        self.one_id = OneLineID(graph, [treatment], [outcome])

        # check the fixability criteria for the treatment
        if len(self.graph.district(treatment).intersection(self.graph.descendants([treatment]))) == 1:
            self.strategy = "a-fixable"
            if self.is_mb_shielded:
                print("\n Treatment is a-fixable and graph is mb-shielded. \n\n Available estimators are:\n \n" +
                      "1. IPW (ipw)\n" +
                      "2. Outcome regression (gformula)\n" +
                      "3. Generalized AIPW (aipw)\n" +
                      "4. Efficient Generalized AIPW (eff-aipw) \n \n" +
                      "Suggested estimator is Efficient Generalized AIPW \n")
            else:
                print("\n Treatment is a-fixable.\n\n Available estimators are :\n \n" +
                      "1. IPW (ipw)\n" +
                      "2. Outcome regression (gformula)\n" +
                      "3. Generalized AIPW (aipw)\n \n" +
                      "Suggested estimator is Generalized AIPW \n")

        elif len(self.graph.district(treatment).intersection(self.graph.children([treatment]))) == 0:
            self.strategy = "p-fixable"
            if self.is_mb_shielded:
                print("\n Treatment is p-fixable and graph is mb-shielded. \n\n Available estimators are:\n\n" +
                      "1. Primal IPW (p-ipw)\n" +
                      "2. Dual IPW (d-ipw)\n" +
                      "3. APIPW (apipw)\n" +
                      "4. Efficient APIPW (eff-apipw) \n \n" +
                      "Suggested estimator is Efficient APIPW \n")
            else:
                print("\n Treatment is p-fixable. \n\n Available estimators are:\n \n" +
                      "1. Primal IPW (p-ipw)\n" +
                      "2. Dual IPW (d-ipw)\n" +
                      "3. APIPW (apipw) \n\n" +
                      "Suggested estimator is APIPW \n")

        elif self.one_id.id():
            self.strategy = "nested-fixable"
            print("\n Effect is identified. \n \n Available estimators:\n \n" +
                  "1. Nested IPW (n-ipw)\n" +
                  "2. Augmented NIPW (anipw) \n\n" +
                  "Suggested estimator is Augmented NIPW \n")

        else:
            self.strategy = "Not ID"
            print("Effect is not identified! \n")

        # finally, fix a valid topological order for estimation queries
        self.p_order = self._find_valid_order("p-fixable") # used for a-fixable/p-fixable strategies
        self.n_order = self._find_valid_order("nested-fixable") # used for nested-fixable strategies
        self.state_space_map_ = {}  # maps from variable names to state space of the variable (binary/continuous)

    def _find_valid_order(self, order_type):
        """
        Utility function to find a valid order that satisfies the requirements in
        Semiparametric Inference (Bhattacharya, Nabi, & Shpitser 2020) paper.

        :param order_type: string specifying if the topological order is required for p-fixable or n-fixable problems.
        :return: list corresponding to a valid topological order.
        """

        # if we are a-fixing or p-fixing, just ensure that the
        # treatment appears after all its non-descendants
        if order_type != "nested-fixable":
            focal_vertices = {self.treatment}

        # otherwise we ensure the vertices in Y* that intersect with district
        # of the treatment appear after all their non-descendants
        else:
            focal_vertices = self.one_id.ystar.intersection(self.graph.district(self.treatment))

        # perform a sort on the nondescendants first
        nondescendants = set(self.graph.vertices) - self.graph.descendants(focal_vertices)
        order = self.graph.subgraph(nondescendants).topological_sort()

        # then sort the focal vertices and its descendants
        order += self.graph.subgraph(self.graph.descendants(focal_vertices)).topological_sort()

        return order

    def _fit_binary_glm(self, data, formula, weights=None):
        """
        Fit a binary GLM to data given a formula.

        :param data: pandas data frame containing the data.
        :param formula: string encoding an R-style formula e.g: Y ~ X1 + X2.
        :return: the fitted model.
        """

        return sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial(), freq_weights=weights).fit()

    def _fit_continuous_glm(self, data, formula, weights=None):
        """
        Fit a linear GLM to data given a formula.

        :param data: pandas data frame containing the data.
        :param formula: string encoding an R-style formula: e.g. Y ~ X1 + X2.
        :return: the fitted model.
        """

        return sm.GLM.from_formula(formula, data=data, family=sm.families.Gaussian(), freq_weights=weights).fit()

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
            raise RuntimeError("IPW will not return valid estimates as treatment is not a-fixable")

        # extract outcome from data frame and compute Markov pillow of treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.p_order)

        if len(mp_T) != 0:
            # fit T | mp(T) and compute probability of treatment for each sample
            formula = self.treatment + " ~ " + '+'.join(mp_T) # + "+ ones"
            model = model_binary(data, formula)
            prob_T = model.predict(data)
        else:
            prob_T = np.ones(len(data)) * np.mean(data[self.treatment])

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
            raise RuntimeError("g-formula will not return valid estimates as treatment is not a-fixable")

        # create a dataset where T=t
        data_assign = data.copy()
        data_assign[self.treatment] = assignment

        # fit Y | T=t, mp(T)
        mp_T = self.graph.markov_pillow([self.treatment], self.p_order)
        if len(mp_T) != 0:
            formula = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T)
            # predict outcome appropriately depending on binary vs continuous
        else:
            formula = self.outcome + " ~ " + self.treatment

        if self.state_space_map_[self.outcome] == "binary":
            model = model_binary(data, formula)
        else:
            model = model_continuous(data, formula)

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
            raise RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")

        # extract the outcome and get Markov pillow of the treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.p_order)

        if len(mp_T) != 0:
            # fit T | mp(T) and predict treatment probabilities
            formula_T = self.treatment + " ~ " + '+'.join(mp_T) #+ "+ ones"
            model = model_binary(data, formula_T)
            prob_T = model.predict(data)
            formula_Y = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T)
        else:
            prob_T = np.ones(len(data)) * np.mean(data[self.treatment])
            formula_Y = self.outcome + " ~ " + self.treatment

        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # fit Y | T=t, mp(T) and predict outcomes under assignment T=t
        data_assign = data.copy()
        data_assign[self.treatment] = assignment
        if self.state_space_map_[self.outcome] == "binary":
            model = model_binary(data, formula_Y)
        else:
            model = model_continuous(data, formula_Y)
        Yhat_vec = model.predict(data_assign)

        # return AIPW estimate
        return np.mean((indices / prob_T) * (Y - Yhat_vec) + Yhat_vec)

    def _eff_augmented_ipw(self, data, assignment, model_binary=None, model_continuous=None):
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
            raise RuntimeError("Augmented IPW will not return valid estimates as treatment is not a-fixable")
        if not self.is_mb_shielded:
            raise RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")

        # extract the outcome and get Markov pillow of treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.p_order)

        # fit T | mp(T) and compute treatment probabilities
        if len(mp_T) != 0:
            formula = self.treatment + " ~ " + '+'.join(mp_T) #+ "+ ones"
            model = model_binary(data, formula)
            prob_T = model.predict(data)
        else:
            prob_T = np.ones(len(data)) * np.mean(data[self.treatment])

        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # compute the primal estimates that we use to compute projections
        primal = (indices / prob_T) * Y
        data["primal"] = primal
        eif_vec = 0

        # get the variables that are actually involved in the efficient influence function
        # TODO: prune extra variables
        eif_vars = [V for V in self.graph.vertices]
        eif_vars.remove(self.treatment)

        # iterate over variables
        for V in eif_vars:

            # get the Markov pillow of the variable
            mpV = self.graph.markov_pillow([V], self.p_order)

            # compute E[primal | mp(V)]
            formula = "primal ~ " + '+'.join(mpV)

            # special case if mp(V) is empty, then E[primal | mp(V)] = E[primal]
            if len(mpV) != 0:
                model_mpV = model_continuous(data, formula)  # primal is a continuous r.v.
                primal_mpV = model_mpV.predict(data)
            else:
                primal_mpV = np.mean(primal)

            # compute E[primal | V, mp(V)]
            formula = formula + "+" + V
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

        # extract the outcome
        Y = data[self.outcome]

        # C := pre-treatment vars and L := post-treatment vars in district of treatment
        C = self.graph.pre([self.treatment], self.p_order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))

        # create copies of the data with treatment assignments T=0 and T=1
        data_T1 = data.copy()
        data_T1[self.treatment] = 1
        data_T0 = data.copy()
        data_T0[self.treatment] = 0

        indices = data[self.treatment] == assignment

        # prob: stores \prod_{Li in L} p(Li | mp(Li))
        # prob_T1: stores \prod_{Li in L} p(Li | mp(Li)) at T=1
        # prob_T0: stores \prod_{Li in L} p(Li | mp(Li)) at T=0

        mp_T = self.graph.markov_pillow([self.treatment], self.p_order)
        indices_T0 = data.index[data[self.treatment] == 0]

        if len(mp_T) != 0:
            formula = self.treatment + " ~ " + '+'.join(mp_T)
            model = model_binary(data, formula)
            prob = model.predict(data)
            prob[indices_T0] = 1 - prob[indices_T0]
            prob_T1 = model.predict(data)
            prob_T0 = 1 - prob_T1
        else:
            prob = np.ones(len(data)) * np.mean(data[self.treatment])
            prob[indices_T0] = 1 - prob[indices_T0]
            prob_T1 = np.ones(len(data)) * np.mean(data[self.treatment])
            prob_T0 = 1 - prob_T1

        # iterate over vertices in L (except the outcome)
        for V in L.difference([self.treatment, self.outcome]):

            # fit V | mp(V)
            mp_V = self.graph.markov_pillow([V], self.p_order)
            formula = V + " ~ " + '+'.join(mp_V)

            # p(V =v | .), p(V = v | . , T=1), p(V = v | ., T=0)
            if self.state_space_map_[V] == "binary":
                model = model_binary(data, formula)
                prob_V = model.predict(data)
                prob_V_T1 = model.predict(data_T1)
                prob_V_T0 = model.predict(data_T0)

                indices_V0 = data.index[data[V] == 0]

                # p(V | .), p(V | ., T=t)
                prob_V[indices_V0] = 1 - prob_V[indices_V0]
                prob_V_T1[indices_V0] = 1 - prob_V_T1[indices_V0]
                prob_V_T0[indices_V0] = 1 - prob_V_T0[indices_V0]

            else:
                model = model_continuous(data, formula)
                E_V = model.predict(data)
                E_V_T1 = model.predict(data_T1)
                E_V_T0 = model.predict(data_T0)

                std = np.std(data[V] - E_V)
                prob_V = norm.pdf(data[V], loc=E_V, scale=std)
                prob_V_T1 = norm.pdf(data[V], loc=E_V_T1, scale=std)
                prob_V_T0 = norm.pdf(data[V], loc=E_V_T0, scale=std)

            prob *= prob_V
            prob_T1 *= prob_V_T1
            prob_T0 *= prob_V_T0

        # special case when the outcome is in L
        if self.outcome in L:

            # fit a binary/continuous model for Y | mp(Y)
            mp_Y = self.graph.markov_pillow([self.outcome], self.p_order)
            formula = self.outcome + " ~ " + '+'.join(mp_Y)
            if self.state_space_map_[self.outcome] == "binary":
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
            raise RuntimeError("Primal IPW will not return valid estimates as treatment is not p-fixable")

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

        # extract the outcome
        Y = data[self.outcome]

        # M := inverse Markov pillow of the treatment
        M = set([m for m in self.graph.vertices if self.treatment in self.graph.markov_pillow([m], self.p_order)])
        M = M.difference(self.graph.district(self.treatment))

        # create a dataset where T=t
        data_assigned = data.copy()
        data_assigned[self.treatment] = assignment

        prob = 1  # stores \prod_{Mi in M} p(Mi | mp(Mi))|T=t / p(Mi | mp(Mi))
        for V in M.difference([self.outcome]):

            # Fit V | mp(V)
            mp_V = self.graph.markov_pillow([V], self.p_order)
            formula = V + " ~ " + '+'.join(mp_V)

            # p(V = 1 | .), p(V = 1 | . , T=assigned)
            if self.state_space_map_[V] == "binary":
                model = model_binary(data, formula)
                prob_V = model.predict(data)
                prob_V_assigned = model.predict(data_assigned)

                indices_V0 = data.index[data[V] == 0]

                # p(V | .) and p(V | ., T=assignment)
                prob_V[indices_V0] = 1 - prob_V[indices_V0]
                prob_V_assigned[indices_V0] = 1 - prob_V_assigned[indices_V0]

            else:
                model = model_continuous(data, formula)
                E_V = model.predict(data)
                E_V_assigned = model.predict(data_assigned)

                std = np.std(data[V] - E_V)
                prob_V = norm.pdf(data[V], loc=E_V, scale=std)
                prob_V_assigned = norm.pdf(data[V], loc=E_V_assigned, scale=std)

            prob *= prob_V_assigned/prob_V

        # special case for if the outcome is in M
        if self.outcome in M:
            mp_Y = self.graph.markov_pillow([self.outcome], self.p_order)
            formula = self.outcome + " ~ " + '+'.join(mp_Y)
            if self.state_space_map_[self.outcome] == "binary":
                model = model_binary(data, formula)
            else:
                model = model_continuous(data, formula)
            Yhat_assigned = model.predict(data_assigned)
        else:
            Yhat_assigned = Y

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
            raise RuntimeError("Dual IPW will not return valid estimates as treatment is not p-fixable")

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
            raise RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")

        # compute the primal and dual estimates and them to the data frame
        beta_primal = self._beta_primal(data, assignment, model_binary, model_continuous)
        beta_dual = self._beta_dual(data, assignment, model_binary, model_continuous)
        data["beta_primal"] = beta_primal
        data["beta_dual"] = beta_dual

        # C := pre-treatment vars
        # L := post-treatment vars in the district of T
        # M := post treatment vars not in L (a.k.a. the rest)
        C = self.graph.pre([self.treatment], self.p_order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L

        IF = 0

        # iterate over all post-treatment variables
        for V in post:

            # compute all predecessors according to the topological order
            pre_V = self.graph.pre([V], self.p_order)

            # if the variables is in M, project using the primal otherwise use the dual
            # to fit E[beta | pre(V)]
            if V in M:
                formula = "beta_primal" + " ~ " + '+'.join(pre_V)
            elif V in L:
                formula = "beta_dual" + " ~ " + '+'.join(pre_V)

            # special logic for if there are no predecessors for the variable (which could only happen to T)
            if len(pre_V) != 0:
                model_preV = model_continuous(data, formula)  # primal/dual is a continuous r.v.
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
            model_C = model_continuous(data, formula)  # dual is a continuous r.v.
            IF += model_C.predict(data)

        # return APIPW estimate
        return np.mean(IF)

    def _eff_augmented_primal_ipw(self, data, assignment, model_binary=None, model_continuous=None):
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
            raise RuntimeError("Augmented primal IPW will not return valid estimates as treatment is not p-fixable")
        if not self.is_mb_shielded:
            raise RuntimeError("EIF will not return valid estimates as graph is not mb-shielded")

        # compute primal and dual estimates and add them to the data frame
        beta_primal = self._beta_primal(data, assignment, model_binary, model_continuous)
        beta_dual = self._beta_dual(data, assignment, model_binary, model_continuous)
        data["beta_primal"] = beta_primal
        data["beta_dual"] = beta_dual

        # C := pre-treatment vars
        # L := post-treatment vars in the district of T
        # M := post treatment vars not in L (a.k.a. the rest)
        C = self.graph.pre([self.treatment], self.p_order)
        post = set(self.graph.vertices).difference(C)
        L = post.intersection(self.graph.district(self.treatment))
        M = post - L

        IF = 0

        # iterate over all variables
        for V in self.graph.vertices:

            # get the Markov pillow
            mp_V = self.graph.markov_pillow([V], self.p_order)

            # if the variables is in M, project using the primal otherwise use the dual
            # to fit E[beta | mp(V)]
            if V in M:
                formula = "beta_primal" + " ~ " + '+'.join(mp_V)
            else:
                formula = "beta_dual" + " ~ " + '+'.join(mp_V)

            # special logic for if there the Markov pillow is empty
            if len(mp_V) != 0:
                model_mpV = model_continuous(data, formula)
                pred_mpV = model_mpV.predict(data)
            else:
                pred_mpV = np.mean(beta_dual)

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

    def _fit_intrinsic_kernel(self, data, district, model_binary=None, model_continuous=None):
        """
        Get estimates of an intrinsic kernel q_D(D|pa(D)) for each sample.

        :param data: pandas data frame containing the data.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: numpy array of estimated probabilities of the kernel for each sample.
        """

        fixing_prob = np.ones(len(data))

        # get parents of the district
        parents = self.graph.parents(district) - district
        # initialize with ancestral margin as fixing descendants are just sums
        G = self.graph.subgraph(self.graph.ancestors(district))

        # we know D is intrinsic because of prior checks
        remaining_vertices = set(G.vertices) - district - parents  # is subtracting parents here ok? think so..

        while not G.fixable(parents)[0]:

            fixed = False

            # at every step try to simplify as much as possible by fixing
            # childless vertices as these correspond to just sums
            # also find a backup fixable vertex in case there are no childless ones
            iter_gen = iter(remaining_vertices)
            i = 0

            while not fixed and i < len(remaining_vertices):
                i += 1
                V = next(iter_gen)
                if len(G.children([V])) == 0:
                    G.fix([V])
                    fixed = True
                    remaining_vertices.remove(V)
                elif G.fixable([V])[0]:
                    fixable_V = V

            # if we fixed something go back and see if we can fix another childless vertex
            # or if the requirement that parents are fixable is satisfied
            if fixed:
                continue

            # otherwise do a true fix corresponding to reweighting
            # using the fixable V that we found
            V = fixable_V
            mb_V = G.markov_blanket([V])
            formula = V + " ~ " + '+'.join(mb_V)

            # p(V = 1 | .)
            if self.state_space_map_[V] == "binary":
                model = model_binary(data, formula, 1/fixing_prob)
                prob_V = model.predict(data)
                indices_V0 = data.index[data[V] == 0]

                # p(V | .)
                prob_V[indices_V0] = 1 - prob_V[indices_V0]

            # handling continuous data
            else:
                model = model_continuous(data, formula, 1/fixing_prob)
                E_V = model.predict(data)
                std = np.std(data[V] - E_V)
                prob_V = norm.pdf(data[V], loc=E_V, scale=std)

            fixing_prob *= prob_V
            G.fix([V])
            remaining_vertices.remove(V)

        # get the Markov blanket of the parents in this final graph
        mb_parents = G.markov_blanket(parents)

        # iterate over each parent and get weights required to fix them
        for V in parents:

            # p(V = 1 | .)
            mp_V = mb_parents.intersection(self.graph.pre([V], self.n_order))
            if len(mp_V) != 0:
                formula = V + " ~ " + '+'.join(mp_V)
            else:
                formula = V + " ~ -1 + 1"

            if self.state_space_map_[V] == "binary":
                model = model_binary(data, formula, 1 / fixing_prob)
                prob_V = model.predict(data)
                indices_V0 = data.index[data[V] == 0]

                # p(V | .)
                prob_V[indices_V0] = 1 - prob_V[indices_V0]

            # handling continuous data
            else:
                model = model_continuous(data, formula, 1 / fixing_prob)
                E_V = model.predict(data)
                std = np.std(data[V] - E_V)
                prob_V = norm.pdf(data[V], loc=E_V, scale=std)

            fixing_prob *= prob_V

        kernel_prob = np.ones(len(data))

        # iterate over member of the intrinsic set and get the final kernel weights
        for V in district:

            # p(V = 1 | .)
            mp_V = parents.union(district.intersection(self.graph.pre(V, self.n_order)))
            if len(mp_V) != 0:
                formula = V + " ~ " + '+'.join(mp_V)
            else:
                formula = V + " ~ -1 + 1"

            if self.state_space_map_[V] == "binary":
                model = model_binary(data, formula, 1 / fixing_prob)
                prob_V = model.predict(data)
                indices_V0 = data.index[data[V] == 0]

                # p(V | .)
                prob_V[indices_V0] = 1 - prob_V[indices_V0]

            # handling continuous data
            else:
                model = model_continuous(data, formula, 1 / fixing_prob)
                E_V = model.predict(data)
                std = np.std(data[V] - E_V)
                prob_V = norm.pdf(data[V], loc=E_V, scale=std)

            kernel_prob *= prob_V

        return kernel_prob

    def _get_nested_rebalanced_weights(self, data, model_binary=None, model_continuous=None):
        """
        Get the rebalancing weights required for nested IPW and augmented nested IPW

        :param data: pandas data frame containing the data.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :return: numpy array corresponding to rebalancing weights.
        """

        # first get all districts of GY* that intersect with district of T
        district_T = self.graph.district(self.treatment)
        modified_districts = [district for district in self.one_id.Gystar.districts
                              if len(district.intersection(district_T)) != 0]

        # placeholder for the rebalancing probability
        rebalance_prob = np.ones(len(data))

        # iterate over each modified district
        for district in modified_districts:

            # first compute weights in the denominator of \prod_{Vi in district} p(Vi | mp(Vi))
            for V in district:

                # Fit V | mp(V)
                mp_V = self.graph.markov_pillow([V], self.n_order)
                formula = V + " ~ " + '+'.join(mp_V)

                # p(V = 1 | .)
                if self.state_space_map_[V] == "binary":
                    model = model_binary(data, formula)
                    prob_V = model.predict(data)
                    indices_V0 = data.index[data[V] == 0]

                    # p(V | .)
                    prob_V[indices_V0] = 1 - prob_V[indices_V0]

                # handling continuous data
                else:
                    model = model_continuous(data, formula)
                    E_V = model.predict(data)
                    std = np.std(data[V] - E_V)
                    prob_V = norm.pdf(data[V], loc=E_V, scale=std)

                rebalance_prob *= 1 / prob_V

            # now compute the q_D(D | pa(D))
            rebalance_prob *= self._fit_intrinsic_kernel(data, district, model_binary, model_continuous)

        return 1/rebalance_prob

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
            raise RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")

        # fit T | mp(T) with the rebalanced weights and compute the nested IPW
        rebalance_weights = self._get_nested_rebalanced_weights(data, model_binary, model_continuous)
        # extract outcome from data frame and compute Markov pillow of treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.n_order)

        if len(mp_T) != 0:
            # fit T | mp(T) and compute probability of treatment for each sample
            formula = self.treatment + " ~ " + '+'.join(mp_T)
            model = model_binary(data, formula, weights=rebalance_weights)
            prob_T = model.predict(data)
        else:
            prob_T = np.ones(len(data)) * np.average(data[self.treatment], weights=rebalance_weights)

        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]

        # compute nested IPW estimate
        indices = data[self.treatment] == assignment
        return np.mean((indices / prob_T) * Y)

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
            raise RuntimeError("Nested IPW will not return valid estimates as causal effect is not identified")

        # get the rebalancing weights
        rebalance_weights = self._get_nested_rebalanced_weights(data, model_binary, model_continuous)

        # extract the outcome and get Markov pillow of the treatment
        Y = data[self.outcome]
        mp_T = self.graph.markov_pillow([self.treatment], self.n_order)

        if len(mp_T) != 0:
            # fit T | mp(T) and predict treatment probabilities
            formula_T = self.treatment + " ~ " + '+'.join(mp_T)  # + "+ ones"
            model = model_binary(data, formula_T, weights=rebalance_weights)
            prob_T = model.predict(data)
            formula_Y = self.outcome + " ~ " + self.treatment + '+' + '+'.join(mp_T)
        else:
            prob_T = np.ones(len(data)) * np.average(data[self.treatment], weights=rebalance_weights)
            formula_Y = self.outcome + " ~ " + self.treatment

        indices_T0 = data.index[data[self.treatment] == 0]
        prob_T[indices_T0] = 1 - prob_T[indices_T0]
        indices = data[self.treatment] == assignment

        # fit Y | T=t, mp(T) and predict outcomes under assignment T=t
        data_assign = data.copy()
        data_assign[self.treatment] = assignment
        if self.state_space_map_[self.outcome] == "binary":
            model = model_binary(data, formula_Y, weights=rebalance_weights)
        else:
            model = model_continuous(data, formula_Y, weights=rebalance_weights)
        Yhat_vec = model.predict(data_assign)

        # return ANIPW estimate
        return np.mean((indices / prob_T) * (Y - Yhat_vec) + Yhat_vec)

    def compute_effect(self, data, estimator, model_binary=None, model_continuous=None, n_bootstraps=0, alpha=0.05):
        """
        Bootstrap functionality to compute the Average Causal Effect if the outcome is continuous
        or the Causal Odds Ratio if the outcome is binary. Returns the point estimate
        as well as lower and upper quantiles for a user specified confidence level.

        :param data: pandas data frame containing the data.
        :param estimator: string indicating what estimator to use: e.g. eff-apipw.
        :param model_binary: string specifying modeling strategy to use for binary variables: e.g. glm-binary.
        :param model_continuous: string specifying modeling strategy to use for continuous variables: e.g. glm-continuous.
        :param n_bootstraps: number of bootstraps.
        :param alpha: the significance level with the default value of 0.05.
        :return: one float corresponding to ACE/OR if n_bootstraps=0, else three floats corresponding to ACE/OR, lower quantile, upper quantile.
        """

        # instantiate modeling strategy with defaults
        if not model_binary:
            model_binary = self._fit_binary_glm
        if not model_continuous:
            model_continuous = self._fit_continuous_glm

        # add a column of ones to fit intercept terms
        data['ones'] = np.ones(len(data))

        # get state space of all variables
        for colname, colvalues in data.iteritems():
            if set([0, 1]).issuperset(colvalues.unique()):
                self.state_space_map_[colname] = "binary"
            else:
                self.state_space_map_[colname] = "continuous"

        # instantiate estimator and get point estimate of ACE
        method = self.estimators[estimator]
        point_estimate_T1 = method(data, 1, model_binary, model_continuous)
        point_estimate_T0 = method(data, 0, model_binary, model_continuous)

        # if Y is binary report log of odds ration, if Y is continuous report ACE
        if self.state_space_map_[self.outcome] == "binary":
            ace = np.log((point_estimate_T1/(1-point_estimate_T1))/(point_estimate_T0/(1-point_estimate_T0)))
        else:
            ace = point_estimate_T1 - point_estimate_T0

        if n_bootstraps > 0:
            ace_vec = []

            Ql = alpha/2
            Qu = 1 - alpha/2

            # iterate over bootstraps
            for iter in range(n_bootstraps):

                # resample the data with replacement
                data_sampled = data.sample(len(data), replace=True)
                data_sampled.reset_index(drop=True, inplace=True)

                # estimate ACE in resampled data
                estimate_T1 = method(data_sampled, 1, model_binary, model_continuous)
                estimate_T0 = method(data_sampled, 0, model_binary, model_continuous)

                # if Y is binary report log of odds ration, if Y is continuous report ACE
                if self.state_space_map_[self.outcome] == "binary":
                    ace_vec.append(np.log((estimate_T1/(1-estimate_T1))/(estimate_T0/(1-estimate_T0))))
                else:
                    ace_vec.append(estimate_T1 - estimate_T0)

            # calculate the quantiles
            quantiles = np.quantile(ace_vec, q=[Ql, Qu])
            q_low = quantiles[0]
            q_up = quantiles[1]
            return ace, q_low, q_up

        return ace
