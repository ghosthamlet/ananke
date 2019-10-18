"""
Class for Linear Gaussian SEMs parametrized
by a matrix B representing regression coefficients
and a matrix Omega representing correlated errors
"""

import functools
import autograd.numpy as anp
from autograd import grad, hessian
from scipy.optimize import minimize
import numpy as np


class LinearGaussianSEM:

    def __init__(self, graph, method="trust-exact"):
        """
        Constructor

        :param graph: graph (ADMG) corresponding to the linear Gaussian SEM.
        :param method: string indiciating optimization method to be used -- "trust-exact" or "BFGS".
        """

        self.graph = graph
        # raise an error if the choice of optimizer is not valid
        if method not in ["BFGS", "trust-exact"]:
            raise ValueError("Invalid choice of method: {}".format(method))
        self.method = method
        # for a linear Gaussian SEM each edge is a parameter + noise for each vertex
        self.n_params = len(graph.di_edges) + len(graph.bi_edges) + len(graph.vertices)
        self._vertex_index_map = {v: i for i, v in enumerate(self.graph.vertices)}
        self.B_adj, self.omega_adj = self._construct_adjacency_matrices()

        self.X_ = None  # data matrix
        self.S_ = None  # sample covariance matrix
        self.B_ = None  # direct edge coefficients
        self.omega_ = None  # correlation of errors

        self.optim_ = None # scipy.minimize return

    def _construct_adjacency_matrices(self):
        """
        Construct adjacency matrices that correspond to the structure
        of the B and omega matrices of the linear SEM.

        :return: B and omega adjacency matrices
        """

        d = len(self.graph.vertices)
        B_adj, omega_adj = np.zeros((d, d), int), np.zeros((d, d), int)

        # set the B adjacency matrix
        for u, v in self.graph.di_edges:

            B_adj[self._vertex_index_map[v], self._vertex_index_map[u]] = 1

        for i in range(d):

            omega_adj[i, i] = 1

        for u, v in self.graph.bi_edges:
            omega_adj[self._vertex_index_map[u], self._vertex_index_map[v]] = 1
            omega_adj[self._vertex_index_map[v], self._vertex_index_map[u]] = 1

        return B_adj, omega_adj

    def _construct_b_omega(self, params):
        """
        To perform unconstrained optimization of a covariance matrix
        we use the log-Cholesky factorization which is unique if diagonal entries are positive.
        This function constructs the B and Omega matrices with the appropriate structure
        based on directed and bidirected edges in the graph.

        :param params: parameters corresponding entries in the B and Omega matrix
        :return: two D x D matrices B and Omega.
        """

        d = self.X_.shape[1]
        L_list, B_list = [],  []
        omega_counter = 0
        b_counter = len(self.graph.vertices) + len(self.graph.bi_edges)

        for i in range(d):
            for j in range(d):

                # check coefficient matrix for non zero values
                if self.B_adj[i, j] == 0:
                    B_list.append(0)
                else:
                    B_list.append(params[b_counter])
                    b_counter += 1

                # check omega matrix for non zero values
                if self.omega_adj[i, j] == 0:
                    L_list.append(0)
                else:
                    # if it's diagonal ensure positivity
                    if i == j:
                        L_list.append(anp.exp(params[omega_counter]))
                        omega_counter += 1
                    # only enter lower triangular entries
                    elif i < j:
                        L_list.append(params[omega_counter])
                        omega_counter += 1
                    else:
                        L_list.append(0)

        # reshape and return matrices
        L = anp.reshape(anp.array(L_list), self.omega_adj.shape)
        B = anp.reshape(anp.array(B_list), self.B_adj.shape)

        # This is required to enforce zero elements in omega
        omega = anp.multiply(anp.dot(L.T, L), self.omega_adj)

        return B, omega

    def _neg_loglikelihood(self, params):
        """
        Internal likelihood function used to fit parameters.

        :param params: a list of parameters in the model.
        :return: a float corresponding to the negative log likelihood.
        """

        n, d = self.X_.shape
        B, omega = self._construct_b_omega(params)
        eye_inv_beta = anp.linalg.inv(anp.eye(d) - B)
        sigma = anp.dot(eye_inv_beta, anp.dot(omega, eye_inv_beta.T))
        likelihood = -(n/2) * (anp.log(anp.linalg.det(sigma)) + anp.trace(anp.dot(anp.linalg.inv(sigma), self.S_)))
        return -likelihood

    def neg_loglikelihood(self, X, weights=None):
        """
        Calculate log-likelihood of the data given the model.

        :param X: a N x M dimensional data matrix.
        :param weights: optional 1d numpy array with weights for each data point
                        (rows with higher weights are given greater importance).
        :return: a float corresponding to the log-likelihood.
        """

        # first check if the model has been fit otherwise throw an error
        if self.B_ is None:
            raise AssertionError("Model must be fit before likelihood can be calculated.")

        # convert the data frame to a raw numpy array
        n, d = X.shape

        # if no weights were given use artificial equal weights
        if weights is None:
            weights = np.ones((n,))

        X_ = np.zeros((n, d))
        for v in self._vertex_index_map:
            X_[:, self._vertex_index_map[v]] = X[v]

        X_ = X_ - np.average(X_, axis=0, weights=weights)  # centre the data
        S_ = np.cov(X_.T, aweights=weights)

        # calculate the likelihood
        eye_inv_beta = np.linalg.inv(np.eye(d) - self.B_)
        sigma = np.dot(eye_inv_beta, np.dot(self.omega_, eye_inv_beta.T))
        return (n/2) * (np.log(np.linalg.det(sigma)) + np.trace(np.dot(np.linalg.inv(sigma), S_)))

    def fit(self, X, weights=None, tol=1e-6, disp=None):
        """
        Fit the model to data via (weighted) maximum likelihood estimation

        :param X: data -- a N x M dimensional pandas data frame.
        :param weights: optional 1d numpy array with weights for each data point
                        (rows with higher weights are given greater importance).
        :return: self.
        """

        # convert the data frame to a raw numpy array
        n, d = X.shape

        # if no weights were given use artificial equal weights
        if weights is None:
            weights = np.ones((n,))

        self.X_ = np.zeros((n, d))
        for v in self._vertex_index_map:
            self.X_[:, self._vertex_index_map[v]] = X[v]
        # self.X_ = self.X_ - np.mean(self.X_, axis=0)  # centre the data
        self.X_ = self.X_ - np.average(self.X_, axis=0, weights=weights)  # centre the data
        self.S_ = np.cov(X.T, aweights=weights)

        likelihood = functools.partial(self._neg_loglikelihood)
        grad_likelihood = grad(likelihood)
        hess_likelihood = hessian(likelihood)

        #initial_guess = anp.full((self.n_params,), np.random.uniform(0, .1))
        initial_guess = anp.full((self.n_params,), 0)

        if self.method == "BFGS":
            optim = minimize(likelihood,
                             x0=initial_guess,
                             method=self.method,
                             jac=grad_likelihood,
                             tol=tol,
                             options={'disp': disp})
        elif self.method == "trust-exact":
            optim = minimize(likelihood,
                             x0=initial_guess,
                             method=self.method,
                             jac=grad_likelihood,
                             hess=hess_likelihood,
                             tol=tol,
                             options={'disp': disp})
        self.optim_ = optim

        self.B_, self.omega_ = self._construct_b_omega(optim.x)
        return self

    def total_effect(self, A, Y):
        """
        Calculate the total causal effect of a set of treatments A on
        a set of outcomes Y.

        :param A: iterable corresponding to variable names that act as treatments.
        :param Y: iterable corresponding to variable names that act as outcomes.
        :return: a float corresponding to the total causal effect.
        """

        directed_paths = self.graph.directed_paths(A, Y)

        # if there are no directed paths, the effect is 0
        if len(directed_paths) == 0:
            return 0

        # otherwise do path analysis
        total_effect = 0
        for path in directed_paths:

            path_effect = 1
            for u, v in path:

                path_effect *= self.B_[self._vertex_index_map[v], self._vertex_index_map[u]]

            total_effect += path_effect

        return total_effect

    def draw(self, direction=None):
        """
        Visualize the graph.

        :return : dot language representation of the graph.
        """

        from graphviz import Digraph
        if self.B_ is None:
            raise AssertionError("Model must be fit before model can be drawn.")
        dot = Digraph()

        # set direction from left to right if that's preferred
        if direction == 'LR':
            dot.graph_attr['rankdir'] = direction

        for v in self.graph.vertices.values():
            dot.node(v.name, shape='plaintext', height='.5', width='.5')

        for parent, child in self.graph.di_edges:
            i, j = self._vertex_index_map[child], self._vertex_index_map[parent]
            dot.edge(parent, child, color='blue', label=str(round(self.B_[i, j], 2)), fontsize="12")
        for sib1, sib2 in self.graph.bi_edges:
            i, j = self._vertex_index_map[sib1], self._vertex_index_map[sib2]
            dot.edge(sib1, sib2, dir='both', color='red', label=str(round(self.omega_[i, j], 2)), fontsize="12")

        return dot
