"""
Class for Linear Gaussian SEMs parametrized
by a matrix B representing regression coefficients
and a matrix Omega representing correlated errors
"""

class LinearGaussianSEM:

    def __init__(self, graph):
        """
        Constructor

        :param graph: graph (ADMG) corresponding to the linear Gaussian SEM
        :param B: matrix of coefficients for the effect of each variable on another (should be lower triangular
                  to maintain acyclicity)
        :param omega: matrix of coefficients representing covariance between error terms of the SEM
                      should be positive definite)
        """

        self.graph = graph
