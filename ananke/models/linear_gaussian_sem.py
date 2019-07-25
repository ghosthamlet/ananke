"""
Class for Linear Gaussian SEMs parametrized
by a matrix B representing regression coefficients
and a matrix Omega representing correlated errors
"""

import functools
import autograd.numpy as anp
from autograd import grad, hessian
from scipy.optimize import minimize

class LinearGaussianSEM:

    def __init__(self, graph, B=None, omega=None):
        """
        Constructor

        :param graph: graph (ADMG) corresponding to the linear Gaussian SEM.
        :param B: matrix of coefficients for the effect of each variable on another (should be lower triangular
                  to maintain acyclicity).
        :param omega: matrix of coefficients representing covariance between error terms of the SEM
                      should be positive definite).
        """

        self.graph = graph
        self.B = B
        self.omega = omega

        # a mapping from vertex to index for easy referencing of matrix entries
        vertex_index = {v: i for (i, v) in enumerate(self.graph.vertices)}

    def fit(self, data):
        """

        :param data: Fit the model to data -- an N x M dimensional matrix.
        :return: None.
        """

        pass

