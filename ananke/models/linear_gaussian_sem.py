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

        # TODO: check the size of these matrices to be d by d + lower triangular B + positive semi-definite omega
        # TODO: throw an exception if not
        self.X = None  # data matrix
        self.S = None  # sample covariance
        self.B = None  # direct edge coefficients
        self.omega = None  # correlation of errors
        self.edge_param_map = {}

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

    def _construct_omega(self, params):
        """
        To perform unconstrained optimization of a covariance matrix
        we use the log-Cholesky factorization which is unique if diagonal entries are positive.
        This function constructs the Omega matrix with the appropriate structure
        based on bidirected edges in the graph.

        :param params: parameters corresponding entries in the Omega matrix
        :return: a D x D matrix Omega.
        """

        d = self.X.shape[1]
        L_list = []
        counter = 0

        for i in range(d):
            for j in range(d):

                if i == j:
                    L_list.append(anp.exp(params[counter]))
                    counter += 1


                else:
                    if i == j:
                        L_list.append(anp.exp(x[omega_struct[i, j] - 1]))
                    elif i < j:
                        L_list.append(x[omega_struct[i, j] - 1])
                    else:
                        L_list.append(x[omega_struct[j, i] - 1])
        L = anp.reshape(anp.array(L_list), omega_struct.shape)
        omega = anp.dot(L.T, L)

        return omega

    def _likelihood(self, params, X):
        """
        Internal
        :return:
        """

        n, d = self.X.shape
        L = anp.zeros((d, d))
        #L[0, 0] = 0
        #L[1, 1] = 0
        #L[2, 2] = 0
        #L[3, 3] = 0
        #L[3][1] = params[4]
        # L = anp.array([[anp.exp(params[0]), 0, 0, 0],
        #                [0, anp.exp(params[1]), 0, 0],
        #                [0, 0, anp.exp(params[2]), 0],
        #                [0, params[4], 0, anp.exp(params[3])]])
        omega = anp.dot(L.T, L)

        B = anp.array([[0, 0, 0, 0],
                       [params[5], 0, 0, 0],
                       [0, params[6], 0, 0],
                       [0, 0, params[7], 0]])
        eye_inv_beta = anp.linalg.inv(anp.eye(d) - B)
        sigma = anp.dot(eye_inv_beta, anp.dot(omega, eye_inv_beta.T))
        likelihood = -(n / 2) * (anp.log(anp.linalg.det(sigma)) + anp.trace(anp.dot(anp.linalg.inv(sigma), self.S)))
        return -likelihood

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

    def fit(self, X, method="BFGS"):
        """

        :param X: Fit the model to X -- a N x M dimensional data matrix.
        :return: None.
        """

        self.X = np.copy(X)
        self.S = np.cov(X.T)

        likelihood = functools.partial(self._likelihood, X=self.X)
        grad_likelihood = grad(likelihood)

        if method == "BFGS":
            optim = minimize(likelihood,
                             x0=anp.full((self.n_params,), 0),
                             method=method,
                             jac=grad_likelihood)
        print(optim.x)


