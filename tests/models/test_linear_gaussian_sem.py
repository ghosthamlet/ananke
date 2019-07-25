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
        self.assertEqual(4, model.n_params)

        # generate data from an ADMG A->B->C->D B<->D and try to fit
        N = 2000
        dim = 4

        omega = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        beta = np.array([[0, 0, 0, 0],
                         [3, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 3, 2.5, 0]])

        true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
        data = np.random.multivariate_normal([0] * dim, true_sigma, size=N)

        # first test that without fitting you can't compute likelihood
        with self.assertRaises(AssertionError):
            model.likelihood(data)

if __name__ == '__main__':
    unittest.main()