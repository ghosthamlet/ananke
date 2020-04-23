
import unittest
import numpy as np
from scipy.special import expit
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ananke.graphs import ADMG
from ananke.estimation import AverageCausalEffect

TOL = 0.1


class TestCounterfactualMean(unittest.TestCase):

    def test_a_fixability(self):

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y', 'D1', 'D2']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y'), ('M', 'D1'), ('Y', 'D2'), ('D1', 'D2')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y'), ('D1', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        data = pd.DataFrame()
        ace = AverageCausalEffect(G, 'T', 'Y')

        self.assertFalse(ace.is_mb_shielded)
        self.assertEqual(ace.strategy, "a-fixable")
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "eff-aipw")

    # a-fixablity with no C and binary/continuous outcome
    def test_ignorable_model(self):
        np.random.seed(0)
        vertices = ['T', 'Y']
        di_edges = [('T', 'Y')]
        bi_edges = []
        G = ADMG(vertices, di_edges, bi_edges)

        size = 5000
        T = np.random.binomial(1, 0.3, size)

        # First dataset
        ace_truth_1 = 0.4
        Y_1 = -0.2 + ace_truth_1*T + np.random.normal(0, 1, size)
        data_1 = pd.DataFrame({'T': T, 'Y': Y_1})

        ace_1 = AverageCausalEffect(G, 'T', 'Y')
        ace_1_ipw, _, _ = ace_1.bootstrap_ace(data_1, "ipw")
        ace_1_gformula, _, _ = ace_1.bootstrap_ace(data_1, "gformula")
        ace_1_aipw, _, _ = ace_1.bootstrap_ace(data_1, "aipw")
        ace_1_eff_aipw, _, _ = ace_1.bootstrap_ace(data_1, "eff-aipw")

        self.assertTrue(ace_1.is_mb_shielded)
        self.assertEqual(ace_1.strategy, "a-fixable")
        self.assertTrue(abs(ace_1_ipw - ace_truth_1) < TOL)
        self.assertTrue(abs(ace_1_gformula - ace_truth_1) < TOL)
        self.assertTrue(abs(ace_1_aipw - ace_truth_1) < TOL)
        self.assertTrue(abs(ace_1_eff_aipw - ace_truth_1) < TOL)

        # Second dataset
        ace_truth_2 = -0.4
        p_y = expit(0.8 + ace_truth_2*T)
        Y_2 = np.random.binomial(1, p_y, size)
        # print(stats.describe(Y_2))
        data_2 = pd.DataFrame({'T': T, 'Y': Y_2})

        ace_2 = AverageCausalEffect(G, 'T', 'Y')
        ace_2_gformula, _, _ = ace_2.bootstrap_ace(data_2, "gformula")
        ace_2_aipw, _, _ = ace_2.bootstrap_ace(data_2, "aipw")

        print("ACE_truth = ", ace_truth_2)
        print(ace_2_gformula)
        print(ace_2_aipw)
        self.assertTrue(abs(ace_2_gformula - ace_truth_2) < TOL)
        self.assertTrue(abs(ace_2_aipw - ace_truth_2) < TOL)

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

        # c1 = f(u3, u4)
        eps_c1 = np.random.normal(0, 1, size)
        C1 = U3 + U4 + eps_c1

        # c2 = f(u5, u6)
        eps_c2 = np.random.normal(0, 1, size)
        C2 = U5 * U6 + eps_c2

        # z1 = f(c1, c2, u1, u2)
        p_z1 = expit(0.4 + 0.3 * C1 - 0.4 * C2 - 0.5 * U1 * U2)
        Z1 = np.random.binomial(1, p_z1, size)

        # z2 = f(z1, u3, u4)
        eps_z2 = np.random.normal(0, 1, size)
        Z2 = 1 + Z1 + U3 + U4 + eps_z2

        # T = f(C1, C2, Z1, U1, U2)
        p_t1 = expit(0.5 - 0.3 * C1 - 0.4 * C2 + 0.3 * U1 - 0.3 * U2)
        T = np.random.binomial(1, p_t1, size)

        # M = f(C1, C2, T)
        eps_m1 = np.random.normal(0, 1, size)
        M = 1 + 0.5 * C1 - 0.8 * C2 - 0.5 * T + eps_m1

        # Y = f(M, U5, U6)
        eps_y = np.random.normal(0, 1, size)
        Y = 1 + 1 * M + U5 + U6 + eps_y

        data = pd.DataFrame({'C1':C1, 'C2':C2, 'Z1':Z1, 'Z2':Z2, 'T':T, 'M':M, 'Y':Y})

        ace_truth = -0.5

        ace = AverageCausalEffect(G, 'T', 'Y')
        ace_ipw, _, _ = ace.bootstrap_ace(data, "ipw")
        ace_gformula, _, _ = ace.bootstrap_ace(data, "gformula")
        ace_aipw, _, _ = ace.bootstrap_ace(data, "aipw")
        ace_eff, _, _ = ace.bootstrap_ace(data, "eff-aipw")

        print(ace_ipw)
        print(ace_gformula)
        print(ace_aipw)
        print(ace_eff)
        self.assertTrue(abs(ace_ipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_gformula - ace_truth) < TOL)
        self.assertTrue(abs(ace_aipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_eff - ace_truth) < TOL)

    def test_p_fixability(self):

        # First graph
        vertices_1 = ['C', 'T', 'M', 'L', 'Y']
        di_edges_1 = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('M', 'Y'), ('L', 'Y')]
        bi_edges_1 = [('T', 'L'), ('T', 'Y')]
        G_1 = ADMG(vertices_1, di_edges_1, bi_edges_1)

        ace_1 = AverageCausalEffect(G_1, 'T', 'Y')
        self.assertTrue(ace_1.is_mb_shielded)
        self.assertEqual(ace_1.strategy, "p-fixable")

        # Second graph
        vertices_2 = ['C', 'T', 'M', 'L', 'Y']
        di_edges_2 = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('T', 'Y'), ('L', 'Y')]
        bi_edges_2 = [('T', 'L'), ('M', 'Y')]
        G_2 = ADMG(vertices_2, di_edges_2, bi_edges_2)

        ace_2 = AverageCausalEffect(G_2, 'T', 'Y')
        self.assertTrue(ace_2.is_mb_shielded)
        self.assertEqual(ace_2.strategy, "p-fixable")

        # Third graph
        vertices_3 = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges_3 = [('C1', 'T'), ('C1', 'L'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges_3 = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G_3 = ADMG(vertices_3, di_edges_3, bi_edges_3)

        data = pd.DataFrame()
        ace_3 = AverageCausalEffect(G_3, 'T', 'Y')

        self.assertFalse(ace_3.is_mb_shielded)
        self.assertEqual(ace_3.strategy, "p-fixable")
        with self.assertRaises(RuntimeError):
            ace_3.bootstrap_ace(data, "eff-apipw")

    def test_p_fixability_compute_ace(self):
        np.random.seed(0)
        vertices = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)

        vertices_hidden = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6']
        di_edges_hidden = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'),
                           ('C2', 'L'), ('C2', 'Y'), ('T', 'M'), ('M', 'L'), ('L', 'Y'),
                           ('U1', 'Z1'), ('U1', 'C1'), ('U2', 'Z1'), ('U2', 'C1'),
                           ('U3', 'C2'), ('U3', 'Z2'), ('U4', 'C2'), ('U4', 'Z2'),
                           ('U5', 'T'), ('U5', 'L'), ('U6', 'T'), ('U6', 'L')]
        G_hidden = ADMG(vertices_hidden, di_edges_hidden, [])

        size = 20000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.6, size)
        U4 = np.random.uniform(-1, 0.2, size)
        U5 = np.random.binomial(1, 0.3, size)
        U6 = np.random.uniform(0.5, 1.5, size)

        # z1 = f(u1, u2)
        p_z1 = expit(0.4 - U1 + U2)
        Z1 = np.random.binomial(1, p_z1, size)

        # c1 = f(z1, u1, u2)
        p_c1 = expit(-0.1 + U1 - U2)  # + 0.5*Z1)
        C1 = np.random.binomial(1, p_c1, size)

        # c2 = f(u3, u4)
        eps_c2 = np.random.normal(0, 1, size)
        C2 = 1 + U3 - U4 + eps_c2

        # z2 = f(u3, u4)
        eps_z2 = np.random.normal(0, 1, size)
        Z2 = -0.5 + U3 - U4 + eps_z2

        # t = f(c1, c2, u5, u6)
        p_t = expit(0.5 + 0.5 * C1 - 0.4 * C2 - 0.4 * U5 + 0.4 * U6)
        T = np.random.binomial(1, p_t, size)

        # m = f(t)
        p_m = expit(-0.3 + 1.5 * T - 0.3 * C2)
        M = np.random.binomial(1, p_m, size)

        # l = f(m, c1, c2, u5, u6)
        p_l = expit(0.75 - 0.8 * M - 0.4 * C1 - 0.3 * C2 - 0.4 * U5 + 0.5 * U6)
        L = np.random.binomial(1, p_l, size)

        # y = f(l)
        eps_y = np.random.normal(0, 1, size)
        Y = 1 + 1 * L + C2 + eps_y

        data = pd.DataFrame({'C1': C1, 'C2': C2, 'Z1': Z1, 'Z2': Z2, 'T': T, 'M': M, 'L': L, 'Y': Y})
        data_hidden = pd.DataFrame({'C1': C1, 'C2': C2, 'Z1': Z1, 'Z2': Z2, 'T': T, 'M': M, 'L': L, 'Y': Y,
                                    'U1': U1, 'U2': U2, 'U3': U3, 'U4': U4, 'U5': U5, 'U6': U6})

        # # compute the effect
        # data['ones'] = np.ones(len(data))
        # formula = "Y ~ T + C1 + C2 + U5 + U6 + ones"
        # model = sm.GLM.from_formula(formula, data=data, family=sm.families.Gaussian()).fit()
        # data_T1 = data.copy()
        # data_T1['T'] = 1
        # data_T0 = data.copy()
        # data_T0['T'] = 0
        # ace_hat = np.mean(model.predict(data_T1) - model.predict(data_T0))

        ace_hidden = AverageCausalEffect(G_hidden, 'T', 'Y')
        ace_hidden_ipw, _, _ = ace_hidden.bootstrap_ace(data_hidden, "ipw")
        ace_hidden_aipw, _, _ = ace_hidden.bootstrap_ace(data_hidden, "aipw")
        ace_hidden_eff, _, _ = ace_hidden.bootstrap_ace(data_hidden, "eff-aipw")

        # ace_truth = -0.07

        ace = AverageCausalEffect(G, 'T', 'Y')
        ace_pipw, _, _ = ace.bootstrap_ace(data, "p-ipw")
        ace_dipw, _, _ = ace.bootstrap_ace(data, "d-ipw")
        ace_apipw, _, _ = ace.bootstrap_ace(data, "apipw")
        ace_eff, _, _ = ace.bootstrap_ace(data, "eff-apipw")

        print(ace_hidden_ipw)
        print(ace_hidden_aipw)
        print(ace_hidden_eff)

        print(ace_pipw)
        print(ace_dipw)
        print(ace_apipw)
        print(ace_eff)


        # self.assertTrue(abs(ace_pipw - ace_truth) < TOL)
        # self.assertTrue(abs(ace_dipw - ace_truth) < TOL)
        # self.assertTrue(abs(ace_apipw - ace_truth) < TOL)
        # self.assertTrue(abs(ace_eff - ace_truth) < TOL)
        # with self.assertRaises(RuntimeError):
        #     ace.bootstrap_ace(data, "ipw")
        # with self.assertRaises(RuntimeError):
        #     ace.bootstrap_ace(data, "gformula")
        # with self.assertRaises(RuntimeError):
        #     ace.bootstrap_ace(data, "aipw")
        # with self.assertRaises(RuntimeError):
        #     ace.bootstrap_ace(data, "eff-aipw")

    def test_nested_fixability(self):
        vertices = ['C1',  'C2',  'T', 'M', 'Z', 'R1', 'R2', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'Y'), ('C2', 'T'), ('C2', 'Y'), ('R2', 'Y'), ('Z', 'T'),
                    ('T', 'R1'), ('T', 'Y'), ('R1', 'M'), ('M', 'Y')]
        bi_edges = [('Z', 'R2'), ('T', 'R2'), ('Z', 'R1'), ('C1', 'M'), ('C1', 'Y'), ('C2', 'M'), ('C2', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        data = pd.DataFrame({'Y': np.ones(100)})

        ace = AverageCausalEffect(G, 'T', 'Y')
        ace_nipw, _, _ = ace.bootstrap_ace(data, "n-ipw")
        ace_anipw, _, _ = ace.bootstrap_ace(data, "anipw")

        self.assertEqual(ace.strategy, "nested-fixable")
        self.assertEqual(ace_nipw, 0)
        self.assertEqual(ace_anipw, 0)
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "p-ipw")
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "d-ipw")
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "apipw")
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "eff-apipw")

    def test_bow_arc(self):
        vertices = ['C', 'T', 'Y']
        di_edges = [('C', 'T'), ('C', 'Y'), ('T', 'Y')]
        bi_edges = [('T', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        data = pd.DataFrame()

        ace = AverageCausalEffect(G, 'T', 'Y')
        self.assertEqual(ace.strategy, "Not ID")
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "n-ipw")
        with self.assertRaises(RuntimeError):
            ace.bootstrap_ace(data, "anipw")

if __name__ == '__main__':
    unittest.main()
