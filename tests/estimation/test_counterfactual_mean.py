
import unittest
import numpy as np
from scipy.special import expit
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ananke.graphs import ADMG
from ananke.estimation import CausalEffect

TOL = 0.1

# print(stats.describe(Y_2))


class TestCounterfactualMean(unittest.TestCase):

    # general check: a-fixablility + mb-shielded ADMG
    def test_a_fixability(self):

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y', 'D1', 'D2']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y'), ('M', 'D1'), ('Y', 'D2'), ('D1', 'D2')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y'), ('D1', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        data = pd.DataFrame()
        ace = CausalEffect(G, 'T', 'Y')

        self.assertFalse(ace.is_mb_shielded)
        self.assertEqual(ace.strategy, "a-fixable")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "eff-aipw")

    # a-fixablity: no C and binary/continuous Y
    def test_ignorable_model(self):
        np.random.seed(0)
        vertices = ['T', 'Y']
        di_edges = [('T', 'Y')]
        bi_edges = []
        G = ADMG(vertices, di_edges, bi_edges)

        size = 5000
        T = np.random.binomial(1, 0.3, size)

        # First dataset: Continuous Y
        ace_truth_1 = 0.4
        Y_1 = -0.2 + ace_truth_1*T + np.random.normal(0, 1, size)
        data_1 = pd.DataFrame({'T': T, 'Y': Y_1})

        ace_1 = CausalEffect(G, 'T', 'Y')
        ace_1_ipw = ace_1.compute_effect(data_1, "ipw")
        ace_1_gformula = ace_1.compute_effect(data_1, "gformula")
        ace_1_aipw = ace_1.compute_effect(data_1, "aipw")
        ace_1_eff_aipw = ace_1.compute_effect(data_1, "eff-aipw")

        self.assertTrue(ace_1.is_mb_shielded)
        self.assertEqual(ace_1.strategy, "a-fixable")
        self.assertTrue(abs(ace_1_ipw - ace_truth_1) < TOL)
        self.assertTrue(abs(ace_1_gformula - ace_truth_1) < TOL)
        self.assertTrue(abs(ace_1_aipw - ace_truth_1) < TOL)
        self.assertTrue(abs(ace_1_eff_aipw - ace_truth_1) < TOL)

        # Second dataset: Binary Y
        ace_truth_2 = -0.4
        p_y = expit(0.8 + ace_truth_2*T)
        Y_2 = np.random.binomial(1, p_y, size)
        data_2 = pd.DataFrame({'T': T, 'Y': Y_2})

        ace_2 = CausalEffect(G, 'T', 'Y')
        ace_2_gformula = ace_2.compute_effect(data_2, "gformula")
        ace_2_aipw = ace_2.compute_effect(data_2, "aipw")

        self.assertTrue(abs(ace_2_gformula - ace_truth_2) < TOL)
        self.assertTrue(abs(ace_2_aipw - ace_truth_2) < TOL)

    # a-fixable: continuous Y
    def test_a_fixability_compute_ace(self):
        np.random.seed(0)

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.6, size)
        U4 = np.random.uniform(-1, 0.2, size)
        U5 = np.random.binomial(1, 0.3, size)
        U6 = np.random.uniform(0.5, 1.5, size)

        C1 = U3 + U4 + np.random.normal(0, 1, size)
        C2 = U5 * U6 + np.random.normal(0, 1, size)

        p_z1 = expit(0.4 + 0.3 * C1 - 0.4 * C2 - 0.5 * U1 * U2)
        Z1 = np.random.binomial(1, p_z1, size)

        Z2 = 1 + Z1 + U3 + U4 + np.random.normal(0, 1, size)

        p_t1 = expit(0.5 - 0.3 * C1 - 0.4 * C2 + 0.3 * U1 - 0.3 * U2)
        T = np.random.binomial(1, p_t1, size)

        M = 1 + 0.5 * C1 - 0.8 * C2 - 0.5 * T + np.random.normal(0, 1, size)
        Y = 1 + 1 * M + U5 + U6 + np.random.normal(0, 1, size)

        data = pd.DataFrame({'C1':C1, 'C2':C2, 'Z1':Z1, 'Z2':Z2, 'T':T, 'M':M, 'Y':Y})

        ace_truth = -0.5

        ace = CausalEffect(G, 'T', 'Y')
        ace_ipw = ace.compute_effect(data, "ipw")
        ace_gformula = ace.compute_effect(data, "gformula")
        ace_aipw = ace.compute_effect(data, "aipw")
        ace_eff = ace.compute_effect(data, "eff-aipw")

        self.assertTrue(abs(ace_ipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_gformula - ace_truth) < TOL)
        self.assertTrue(abs(ace_aipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_eff - ace_truth) < TOL)

    # general check: p-fixablility + mb-shielded ADMG
    def test_p_fixability(self):

        # First graph: mb-shielded
        vertices_1 = ['C', 'T', 'M', 'L', 'Y']
        di_edges_1 = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('M', 'Y'), ('L', 'Y')]
        bi_edges_1 = [('T', 'L'), ('T', 'Y')]
        G_1 = ADMG(vertices_1, di_edges_1, bi_edges_1)

        ace_1 = CausalEffect(G_1, 'T', 'Y')
        self.assertTrue(ace_1.is_mb_shielded)
        self.assertEqual(ace_1.strategy, "p-fixable")

        # Second graph: mb-shielded
        vertices_2 = ['C', 'T', 'M', 'L', 'Y']
        di_edges_2 = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('T', 'Y'), ('L', 'Y')]
        bi_edges_2 = [('T', 'L'), ('M', 'Y')]
        G_2 = ADMG(vertices_2, di_edges_2, bi_edges_2)

        ace_2 = CausalEffect(G_2, 'T', 'Y')
        self.assertTrue(ace_2.is_mb_shielded)
        self.assertEqual(ace_2.strategy, "p-fixable")

        # Third graph: not mb-shielded
        vertices_3 = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges_3 = [('C1', 'T'), ('C1', 'L'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges_3 = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G_3 = ADMG(vertices_3, di_edges_3, bi_edges_3)

        data = pd.DataFrame()
        ace_3 = CausalEffect(G_3, 'T', 'Y')

        self.assertFalse(ace_3.is_mb_shielded)
        self.assertEqual(ace_3.strategy, "p-fixable")
        with self.assertRaises(RuntimeError):
            ace_3.compute_effect(data, "eff-apipw")

    # p-fixable: Binary M, Binary L, Continuous Y \not\in {L, M}
    def test_p_fixability_binaryML(self):
        np.random.seed(0)
        vertices = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 1000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.6, size)
        U4 = np.random.uniform(-1, 0.2, size)
        U5 = np.random.binomial(1, 0.3, size)
        U6 = np.random.uniform(0.5, 1.5, size)

        p_z1 = expit(0.4 - U1 + U2)
        Z1 = np.random.binomial(1, p_z1, size)

        p_c1 = expit(-0.1 + U1 - U2)  # + 0.5*Z1)
        C1 = np.random.binomial(1, p_c1, size)

        C2 = 1 + U3 - U4 + np.random.normal(0, 1, size)

        Z2 = -0.5 + U3 - U4 + np.random.normal(0, 1, size)

        p_t = expit(0.5 + 0.5 * C1 - 0.4 * C2 - 0.4 * U5 + 0.4 * U6)
        T = np.random.binomial(1, p_t, size)

        p_m = expit(-0.3 + 1.5 * T - 0.3 * C2)
        M = np.random.binomial(1, p_m, size)

        p_l = expit(0.75 - 0.8 * M - 0.4 * C1 - 0.3 * C2 - 0.4 * U5 + 0.5 * U6)
        L = np.random.binomial(1, p_l, size)

        Y = 1 + 1 * L + C2 + np.random.normal(0, 1, size)

        data = pd.DataFrame({'C1': C1, 'C2': C2, 'Z1': Z1, 'Z2': Z2, 'T': T, 'M': M, 'L': L, 'Y': Y})

        # Compute the true ACE
        # vertices_hidden = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6']
        # di_edges_hidden = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'),
        #                    ('C2', 'L'), ('C2', 'Y'), ('T', 'M'), ('M', 'L'), ('L', 'Y'),
        #                    ('U1', 'Z1'), ('U1', 'C1'), ('U2', 'Z1'), ('U2', 'C1'),
        #                    ('U3', 'C2'), ('U3', 'Z2'), ('U4', 'C2'), ('U4', 'Z2'),
        #                    ('U5', 'T'), ('U5', 'L'), ('U6', 'T'), ('U6', 'L')]
        # G_hidden = ADMG(vertices_hidden, di_edges_hidden, [])
        # data_hidden = pd.DataFrame({'C1': C1, 'C2': C2, 'Z1': Z1, 'Z2': Z2, 'T': T, 'M': M, 'L': L, 'Y': Y,
        #                             'U1': U1, 'U2': U2, 'U3': U3, 'U4': U4, 'U5': U5, 'U6': U6})
        # ace_hidden = AverageCausalEffect(G_hidden, 'T', 'Y')
        # ace_hidden_ipw, _, _ = ace_hidden.bootstrap_ace(data_hidden, "ipw")
        # ace_hidden_gformula, _, _ = ace_hidden.bootstrap_ace(data_hidden, "gformula")
        # ace_hidden_aipw, _, _ = ace_hidden.bootstrap_ace(data_hidden, "aipw")
        # ace_hidden_eff, _, _ = ace_hidden.bootstrap_ace(data_hidden, "eff-aipw")
        #
        # print(ace_hidden_ipw)
        # print(ace_hidden_gformula)
        # print(ace_hidden_aipw)
        # print(ace_hidden_eff, "\n")
        ace_truth = -0.07

        ace = CausalEffect(G, 'T', 'Y')
        ace_pipw = ace.compute_effect(data, "p-ipw")
        ace_dipw = ace.compute_effect(data, "d-ipw")
        ace_apipw = ace.compute_effect(data, "apipw")
        ace_eff = ace.compute_effect(data, "eff-apipw")

        self.assertTrue(abs(ace_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_dipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_apipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_eff - ace_truth) < TOL)
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "ipw")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "gformula")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "aipw")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "eff-aipw")

    # p-fixable: Continuous M, Continuous L, continuous Y \not\in {L, M}
    def test_p_fixability_continuousML(self):
        np.random.seed(0)
        vertices = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 5000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.6, size)
        U4 = np.random.uniform(-1, 0.2, size)
        U5 = np.random.binomial(1, 0.3, size)
        U6 = np.random.uniform(0.5, 1.5, size)

        p_z1 = expit(0.4 - U1 + U2)
        Z1 = np.random.binomial(1, p_z1, size)

        p_c1 = expit(-0.1 + U1 - U2)
        C1 = np.random.binomial(1, p_c1, size)

        C2 = 1 + U3 - U4 + np.random.normal(0, 1, size)
        Z2 = -0.5 + U3 - U4 + np.random.normal(0, 1, size)

        p_t = expit(0.5 + 0.5 * C1 - 0.4 * C2 - 0.4 * U5 + 0.4 * U6)
        T = np.random.binomial(1, p_t, size)

        M = -0.3 + 0.8 * T - 0.3 * C2 + np.random.normal(0, 1, size)
        L = 0.75 - 1.5 * M - 0.4 * C1 - 0.3 * C2 - 0.4 * U5 + 0.5 * U6 + np.random.normal(0, 1, size)
        Y = 1 + 1 * L + C2 + np.random.normal(0, 1, size)

        data = pd.DataFrame({'C1': C1, 'C2': C2, 'Z1': Z1, 'Z2': Z2, 'T': T, 'M': M, 'L': L, 'Y': Y})

        # Compute the true ACE = (coefficient of T in M)*(coefficient of M in L)
        ace_truth = -1.2

        ace = CausalEffect(G, 'T', 'Y')
        ace_pipw = ace.compute_effect(data, "p-ipw")
        ace_dipw = ace.compute_effect(data, "d-ipw")

        print(ace_pipw)
        print(ace_dipw)

        self.assertTrue(abs(ace_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_dipw - ace_truth) < TOL)

    # p-fixable: Continuous M, no C , continuous Y \in {L} AND continuous Y \in {M}
    def test_p_fixable_continuousY_inML(self):
        np.random.seed(0)
        vertices = ['T', 'M', 'Y']
        di_edges = [('T', 'M'), ('M', 'Y')]
        bi_edges_1 = [('T', 'Y')]
        bi_edges_2 = [('M', 'Y')]

        # First graph: Y \in L (front-door), mb-shielded
        G_1 = ADMG(vertices, di_edges, bi_edges_1)

        # Second graph: Y \in M, not mb-shielded
        G_2 = ADMG(vertices, di_edges, bi_edges_2)

        size = 5000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)

        p_t = expit(0.8 + 0.4 * U1 - 0.8 * U2)
        T = np.random.binomial(1, p_t, size)

        M = -0.3 + 0.8 * T + np.random.normal(0, 1, size)
        Y = 1 + 1 * M + 0.5 * U1 + 0.3 * U2 + np.random.normal(0, 1, size)

        data = pd.DataFrame({'T': T, 'M': M, 'Y': Y})

        # Compute the true ACE = (coefficient of T in M)*(coefficient of M in L)
        ace_truth = 0.8

        # Test on the first graph
        ace_1 = CausalEffect(G_1, 'T', 'Y')
        ace_1_pipw, _, _ = ace_1.compute_effect(data, "p-ipw", n_bootstraps=1)
        ace_1_dipw = ace_1.compute_effect(data, "d-ipw")
        ace_1_apipw = ace_1.compute_effect(data, "apipw")
        ace_1_eff = ace_1.compute_effect(data, "eff-apipw")

        self.assertTrue(ace_1.is_mb_shielded)
        self.assertTrue(abs(ace_1_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_1_dipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_1_apipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_1_eff - ace_truth) < TOL)

        # Test on the second graph
        ace_2 = CausalEffect(G_2, 'T', 'Y')
        ace_2_pipw = ace_2.compute_effect(data, "p-ipw")
        ace_2_dipw = ace_2.compute_effect(data, "d-ipw")
        ace_2_apipw = ace_2.compute_effect(data, "apipw")

        self.assertFalse(ace_2.is_mb_shielded)
        self.assertTrue(abs(ace_2_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_2_dipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_2_apipw - ace_truth) < TOL)

    # p-fixable: Continuous M, no C , binary Y \in {L} AND binary Y \in {M}
    def test_p_fixable_binaryY_inML(self):
        np.random.seed(0)
        vertices = ['T', 'M', 'Y']
        di_edges = [('T', 'M'), ('M', 'Y')]
        bi_edges_1 = [('T', 'Y')]
        bi_edges_2 = [('M', 'Y')]

        # First graph: Y \in L (front-door)
        G_1 = ADMG(vertices, di_edges, bi_edges_1)

        # Second graph: Y \in M
        G_2 = ADMG(vertices, di_edges, bi_edges_2)

        size = 5000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)

        p_t = expit(0.5 - 0.4 * U1 + 0.4 * U2)
        T = np.random.binomial(1, p_t, size)

        M = -0.3 + 0.8 * T + np.random.normal(0, 1, size)

        p_y = expit(-1 + 0.4 * M + 0.5 * U1 + 0.3 * U2)
        Y = np.random.binomial(1, p_y, size)

        data = pd.DataFrame({'T': T, 'M': M, 'Y': Y})

        # Compute the true ACE (log of odds ratio)
        ace_truth = 0.32

        # Test the first graph
        ace_1 = CausalEffect(G_1, 'T', 'Y')
        ace_1_pipw = ace_1.compute_effect(data, "p-ipw")
        ace_1_dipw = ace_1.compute_effect(data, "d-ipw")

        self.assertTrue(abs(ace_1_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_1_dipw - ace_truth) < TOL)

        # Test the second graph
        ace_2 = CausalEffect(G_2, 'T', 'Y')
        ace_2_pipw, _, _ = ace_2.compute_effect(data, "p-ipw", n_bootstraps=1)
        ace_2_dipw = ace_2.compute_effect(data, "d-ipw")

        self.assertTrue(abs(ace_2_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_2_dipw - ace_truth) < TOL)

    # nested-fixable
    def test_nested_fixability(self):
        np.random.seed(0)
        vertices = ['C1',  'C2',  'T', 'M', 'Z', 'R1', 'R2', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'Y'), ('C2', 'T'), ('C2', 'Y'), ('R2', 'Y'), ('Z', 'T'),
                    ('T', 'R1'), ('T', 'Y'), ('R1', 'M'), ('M', 'Y')]
        bi_edges = [('Z', 'R2'), ('T', 'R2'), ('Z', 'R1'), ('C1', 'M'), ('C1', 'Y'), ('C2', 'M'), ('C2', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 0.5, size)
        U3 = np.random.binomial(1, 0.3, size)
        U4 = np.random.uniform(0, 0.5, size)
        U5 = np.random.binomial(1, 0.4, size)
        U6 = np.random.uniform(0, 1.5, size)
        U7 = np.random.binomial(1, 0.3, size)
        U8 = np.random.uniform(0, 0.5, size)
        U9 = np.random.binomial(1, 0.3, size)
        U10 = np.random.uniform(0, 0.5, size)

        p_r2 = expit(-0.2 + U1 - 0.8 * U2 + U3 + U4)
        R2 = np.random.binomial(1, p_r2, size)

        C1 = U7 - U8 + U9 + U10 + np.random.normal(0, 1, size)
        C2 = U7 - U8 + U9 * U10 + np.random.normal(0, 1, size)
        Z = U1 - U2 - U5 - U6 + np.random.normal(0, 1, size)

        p_t1 = expit(0.8 - 0.5 * C1 + 0.5 * C2 + 0.3 * Z + 0.5 * U3 - 0.4 * U4)
        T = np.random.binomial(1, p_t1, size)

        p_r1 = expit(0.2 + 0.7 * T - 0.6 * U5 - 0.6 * U6)
        R1 = np.random.binomial(1, p_r1, size)

        M = 1 - R1 - U7 + U8 + np.random.normal(0, 1, size)
        Y = 1 + R2 + M + 1 * T + C1 + C2 + U9 + U10 + np.random.normal(0, 1, size)

        data = pd.DataFrame({'C1': C1, 'C2': C2, 'R1': R1, 'R2': R2, 'Z': Z, 'T': T, 'M': M, 'Y': Y})

        ace_truth = 0.8

        ace = CausalEffect(G, 'T', 'Y')
        ace_nipw = ace.compute_effect(data, "n-ipw")
        ace_anipw = ace.compute_effect(data, "anipw")

        self.assertEqual(ace.strategy, "nested-fixable")
        self.assertTrue(abs(ace_nipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_anipw - ace_truth) < TOL)

        # ensure other methods don't work for this graph
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "p-ipw")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "d-ipw")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "apipw")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "eff-apipw")

    def test_nested_fixable_Y_in_DT(self):
        np.random.seed(0)
        vertices = ['C', 'Z1', 'Z2', 'T', 'Y']
        di_edges = [('C', 'T'), ('C', 'Y'), ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'Y')]
        bi_edges = [('Z1', 'T'), ('Z1', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.3, size)
        U4 = np.random.uniform(0, 0.5, size)

        C = np.random.normal(0, 1, size)
        Z1 = U1 - U2 + 0.5 * U3 + U4 + np.random.normal(1, 1, size)

        p_z2 = expit(0.4 - 0.4 * Z1)
        Z2 = np.random.binomial(1, p_z2, size)

        p_t = expit(0.2 + 0.2 * C + 0.5 * Z2 + U1 - U2)
        T = np.random.binomial(1, p_t, size)

        Y = 1 + T - 0.6 * C + U3 + U4 + np.random.normal(0, 1, size)

        ace_truth = 1

        ace = CausalEffect(G, 'T', 'Y')
        data = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'C': C, 'T': T, 'Y': Y})
        ace_nipw = ace.compute_effect(data, "n-ipw")
        ace_anipw = ace.compute_effect(data, "anipw")

        self.assertTrue(abs(ace_nipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_anipw - ace_truth) < TOL)

    # nested-fixablity: continuous variables
    def test_nested_fixable_Y_in_DT_continuousZ(self):
        np.random.seed(0)
        vertices = ['C', 'Z1', 'Z2', 'T', 'Y']
        di_edges = [('C', 'T'), ('C', 'Y'), ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'Y')]
        bi_edges = [('Z1', 'T'), ('Z1', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.3, size)
        U4 = np.random.uniform(0, 0.5, size)

        C = np.random.normal(0, 1, size)
        Z1 = U1 - U2 + 0.5 * U3 + U4 + np.random.normal(1, 1, size)
        Z2 = 0.4 - 0.4 * Z1 + np.random.normal(0, 1, size)

        p_t = expit(0.2 + 0.2 * C + 0.5 * Z2 + U1 - U2)
        T = np.random.binomial(1, p_t, size)

        Y = 1 + T - 0.6 * C + U3 + U4 + np.random.normal(0, 1, size)

        data = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'C': C, 'T': T, 'Y': Y})

        ace_truth = 1

        ace = CausalEffect(G, 'T', 'Y')
        ace.n_order = ['C', 'Z1', 'Z2', 'T', 'Y' ]
        ace_nipw = ace.compute_effect(data, "n-ipw")
        ace_anipw = ace.compute_effect(data, "anipw")

        self.assertTrue(abs(ace_nipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_anipw - ace_truth) < TOL)

    # nested-fixablity
    def test_nested_fixable_conditional_ignorability(self):
        np.random.seed(0)
        vertices = ['C', 'T', 'Y']
        di_edges = [('C', 'Y'), ('T', 'Y')]
        bi_edges = [('C', 'T')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)

        C = -1 + 0.4 * U1 + 0.8 * U2 + np.random.normal(0, 1, size)

        p_t = expit(0.2 + 0.2 * U1 + 0.5 * U2)
        T = np.random.binomial(1, p_t, size)

        p_y = expit(1 + 1.5 * T - 0.6 * C)
        Y = np.random.binomial(1, p_y, size)

        data = pd.DataFrame({'C': C, 'T': T, 'Y': Y})

        ace_truth = 1.5

        ace = CausalEffect(G, 'T', 'Y')
        ace_nipw = ace.compute_effect(data, "n-ipw")
        ace_anipw = ace.compute_effect(data, "anipw")

        self.assertTrue(abs(ace_nipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_anipw - ace_truth) < TOL)

    # not ID
    def test_bow_arc(self):
        vertices = ['C', 'T', 'Y']
        di_edges = [('C', 'T'), ('C', 'Y'), ('T', 'Y')]
        bi_edges = [('T', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        data = pd.DataFrame()

        ace = CausalEffect(G, 'T', 'Y')
        self.assertEqual(ace.strategy, "Not ID")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "n-ipw")
        with self.assertRaises(RuntimeError):
            ace.compute_effect(data, "anipw")


if __name__ == '__main__':
    unittest.main()
