import unittest

from ananke.graphs import ADMG
from ananke.models import LinearGaussianSEM
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose


class TestLinearGaussianSEM(unittest.TestCase):

    def test_model_creation(self):

        vertices = ["A", "B", "C", "D"]
        di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
        bi_edges = [("B", "D")]
        G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        model = LinearGaussianSEM(G)
        self.assertEqual(8, model.n_params)

    def test_adjacency_matrices(self):

        vertices = ["A", "B", "C", "D"]
        di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
        bi_edges = [("B", "D")]
        G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        model = LinearGaussianSEM(G)

        omega = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0.8],
                          [0, 0, 1, 0],
                          [0, 0.8, 0, 1]])

        beta = np.array([[0, 0, 0, 0],
                         [3, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 2.5, 0]])

        # test the adjacency matrices have correct structure
        beta_nonzero = np.array(beta != 0, int)
        self.assertTrue(np.array_equal(beta_nonzero, model.B_adj))
        omega_nonzero = np.array(omega != 0, int)
        self.assertTrue(np.array_equal(omega_nonzero, model.omega_adj))

    def test_verma_model(self):

        vertices = ["A", "B", "C", "D"]
        di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
        bi_edges = [("B", "D")]
        G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        model = LinearGaussianSEM(G)

        # generate data from an ADMG A->B->C->D B<->D and try to fit
        N = 50000
        dim = 4

        omega = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0.8],
                          [0, 0, 1, 0],
                          [0, 0.8, 0, 1]])

        beta = np.array([[0, 0, 0, 0],
                         [3, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 2.5, 0]])

        # generate data
        true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
        X = np.random.multivariate_normal([0] * dim, true_sigma, size=N)
        data = pd.DataFrame({"A": X[:, 0], "B": X[:, 1], "C": X[:, 2], "D": X[:, 3]})

        # test that without fitting you can't compute likelihood
        with self.assertRaises(AssertionError):
            model.neg_loglikelihood(data)

        # test that without fitting you can't draw the model
        with self.assertRaises(AssertionError):
            model.draw()

        # test that invalid optimizer throws an error
        with self.assertRaises(ValueError):
            model = LinearGaussianSEM(G, method="FakeOptimizer")

        # try with BFGS
        model = LinearGaussianSEM(G, method="BFGS")
        model.fit(data)
        self.assertTrue(np.allclose(beta, model.B_, rtol=0.05))
        self.assertTrue(np.allclose(omega, model.omega_, rtol=0.05))

        # try with trust-exact (the default)
        model = LinearGaussianSEM(G, method="trust-exact")
        model.fit(data)
        self.assertTrue(np.allclose(beta, model.B_, rtol=0.1))
        self.assertTrue(np.allclose(omega, model.omega_, rtol=0.1))

        model.neg_loglikelihood(data)

        # computation of causal effects
        self.assertEqual(0, model.total_effect(["D"], ["A"]))
        self.assertAlmostEqual(-7.5, model.total_effect(["A"], ["D"]), delta=0.5)

        model.draw(direction='LR')


    def test_that_omega_decomposition_correct(self):
        """
        In order for the model to be faithful to the graph it is necessary to enforce
        the constraint that the omega matrix contains the correct zeros.

        It is not known why the function fails to converge
        :return:
        """
        N = 2000
        dim = 4


        beta = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, -1.9, 0, 0],
                         [0, 0, 1, 0]])
        omega = np.array([[1.39078604, 0.15565348, 0.1193632, 0.],
               [0.15565348, 1.09943693, 0., 0.01106629],
               [0.1193632, 0., 1.39414306, 0.],
               [0., 0.01106629, 0., 0.93586337]])

        true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
        X = np.random.multivariate_normal([0] * dim, true_sigma, size=N)
        data = pd.DataFrame({"A": X[:, 0], "B": X[:, 1], "C": X[:, 2], "D": X[:, 3]})

        vertices = ['A', 'B', 'C', 'D']
        di_edges = [("B", "C"), ("C", "D")]
        bi_edges = [('A', 'C'), ('A', 'B'), ('B', 'D')]
        G3 = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        model3 = LinearGaussianSEM(G3, method="trust-exact")
        model3.fit(data, tol=1e-10, disp=1000)


        # We are only interested in the zeros of the omega matrix (corresponding to lack of bidirected edges in graph)
        truth_mask = omega == 0
        result_mask = model3.omega_ == 0

        assert_allclose(truth_mask, result_mask)



if __name__ == '__main__':
    unittest.main()