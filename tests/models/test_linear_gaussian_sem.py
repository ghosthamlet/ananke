import unittest

from ananke.graphs import ADMG
from ananke.models import LinearGaussianSEM
import numpy as np


class TestLinearGaussianSEM(unittest.TestCase):

    def test_verma_model(self):

        vertices = ["A", "B", "C", "D"]
        di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
        bi_edges = [("B", "D")]
        G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        model = LinearGaussianSEM(G)
        self.assertEqual(8, model.n_params)

        # generate data from an ADMG A->B->C->D B<->D and try to fit
        N = 5000
        dim = 4

        omega = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0.8],
                          [0, 0, 1, 0],
                          [0, 0.8, 0, 1]])

        beta = np.array([[0, 0, 0, 0],
                         [3, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 2.5, 0]])

        # test the adjacency matrices
        beta_nonzero = np.array(beta != 0, int)
        self.assertTrue(np.array_equal(beta_nonzero, model.B_adj))
        omega_nonzero = np.array(omega != 0, int)
        self.assertTrue(np.array_equal(omega_nonzero, model.omega_adj))

        # generate data
        true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
        data = np.random.multivariate_normal([0] * dim, true_sigma, size=N)

        # test that without fitting you can't compute likelihood
        with self.assertRaises(AssertionError):
            model.likelihood(data)

        # try with BFGS
        model.fit(data, method="BFGS")
        self.assertTrue(np.allclose(beta, model.B, rtol=0.05))
        self.assertTrue(np.allclose(omega, model.omega, rtol=0.05))

        # try with trust-exact (the default)
        model.fit(data)
        self.assertTrue(np.allclose(beta, model.B, rtol=0.05))
        self.assertTrue(np.allclose(omega, model.omega, rtol=0.05))

        model.likelihood(data)

        # computation of causal effects
        self.assertEqual(0, model.total_effect(["D"], ["A"]))
        self.assertAlmostEqual(-7.5, model.total_effect(["A"], ["D"]), delta=0.5)


if __name__ == '__main__':
    unittest.main()