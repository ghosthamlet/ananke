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
import copy


class LinearGaussianSEM:

    def __init__(self, graph):
        """
        Constructor

        :param graph: graph (ADMG) corresponding to the linear Gaussian SEM.
        """

        self.graph = graph
        # for a linear Gaussian SEM each edge is a parameter
        self.n_params = len(graph.di_edges) + len(graph.bi_edges)

        # TODO: check the size of these matrices to be d by d + lower triangular B + positive semi-definite omega
        # TODO: throw an exception if not
        self.X = None  # data matrix
        self.S = None  # sample covariance
        self.B = None  # direct edge coefficients
        self.omega = None  # correlation of errors
        self.vertex_index_map = {}
        self.edge_param_map = {}

    def likelihood(self, X):
        """
        Calculate likelihood of the data given the model.

        :param X: a N x M dimensional data matrix.
        :return: a float corresponding to the likelihood.
        """

        # first check if the model has been fit otherwise throw an error
        if not self.B:
            raise AssertionError("Model must be fit before likelihood can be calculated.")

        n, d = X.shape
        S = np.cov(X.T)
        eye_inv_beta = np.linalg.inv(np.eye(d) - self.B)
        sigma = np.dot(eye_inv_beta, np.dot(self.omega, eye_inv_beta.T))
        return -(n/2) * (np.log(np.linalg.det(sigma)) - np.trace(np.dot(np.linalg.inv(sigma), S)))

    def fit(self, X):
        """

        :param X: Fit the model to X -- a N x M dimensional data matrix.
        :return: None.
        """

        self.X = np.copy(X)
        self.S = np.cov(X.T)

