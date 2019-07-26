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

    def __init__(self, graph):
        """
        Constructor

        :param graph: graph (ADMG) corresponding to the linear Gaussian SEM.
        """

        self.graph = graph
        # for a linear Gaussian SEM each edge is a parameter + noise for each vertex
        self.n_params = len(graph.di_edges) + len(graph.bi_edges) + len(graph.vertices)
        self.vertex_index_map = {v: i for i, v in enumerate(self.graph.vertices)}
        self.B_adj, self.omega_adj = self._construct_adjacency_matrices()

        self.X = None  # data matrix
        self.S = None  # sample covariance
        self.B = None  # direct edge coefficients
        self.omega = None  # correlation of errors

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

            B_adj[self.vertex_index_map[v], self.vertex_index_map[u]] = 1

        for i in range(d):

            omega_adj[i, i] = 1

        for u, v in self.graph.bi_edges:
            omega_adj[self.vertex_index_map[u], self.vertex_index_map[v]] = 1
            omega_adj[self.vertex_index_map[v], self.vertex_index_map[u]] = 1

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

        d = self.X.shape[1]
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

        return B, anp.dot(L.T, L)

    def _likelihood(self, params):
        """
        Internal likelihood function used to fit parameters.

        :param params: a list of parameters in the model.
        :return: a float corresponding to the negative log likelihood.
        """

        n, d = self.X.shape
        B, omega = self._construct_b_omega(params)
        eye_inv_beta = anp.linalg.inv(anp.eye(d) - B)
        sigma = anp.dot(eye_inv_beta, anp.dot(omega, eye_inv_beta.T))
        likelihood = -(n / 2) * (anp.log(anp.linalg.det(sigma)) + anp.trace(anp.dot(anp.linalg.inv(sigma), self.S)))
        return -likelihood

    def likelihood(self, X):
        """
        Calculate log-likelihood of the data given the model.

        :param X: a N x M dimensional data matrix.
        :return: a float corresponding to the log-likelihood.
        """

        # first check if the model has been fit otherwise throw an error
        if self.B is None:
            raise AssertionError("Model must be fit before likelihood can be calculated.")

        n, d = X.shape
        S = np.cov(X.T)
        eye_inv_beta = np.linalg.inv(np.eye(d) - self.B)
        sigma = np.dot(eye_inv_beta, np.dot(self.omega, eye_inv_beta.T))
        return -(n/2) * (np.log(np.linalg.det(sigma)) + np.trace(np.dot(np.linalg.inv(sigma), S)))

    def fit(self, X, method="trust-exact"):
        """

        :param X: Fit the model to X -- a N x M dimensional data matrix.
        :return: None.
        """

        self.X = X - np.mean(X, axis=0) # centre the data
        self.S = np.cov(X.T)

        likelihood = functools.partial(self._likelihood)
        grad_likelihood = grad(likelihood)
        hess_likelihood = hessian(likelihood)

        if method == "BFGS":
            optim = minimize(likelihood,
                             x0=anp.full((self.n_params,), 0),
                             method=method,
                             jac=grad_likelihood)
        elif method == 'trust-exact':
            optim = minimize(likelihood,
                             x0=anp.full((self.n_params,), 0),
                             method="trust-exact",
                             jac=grad_likelihood,
                             hess=hess_likelihood)

        self.B, self.omega = self._construct_b_omega(optim.x)

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

                path_effect *= self.B[self.vertex_index_map[v], self.vertex_index_map[u]]

            total_effect += path_effect

        return total_effect
