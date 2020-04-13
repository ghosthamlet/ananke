
import unittest
import numpy as np
from scipy.special import expit
import pandas as pd
from scipy import stats

from ananke.graphs import ADMG
from ananke.estimation import AverageCausalEffect

TOL = 0.05


class TestCounterfactualMean(unittest.TestCase):

    def test_a_fixability_mbshielded(self):
        np.random.seed(0)

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y', 'D1', 'D2']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'), ('C2', 'D1'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y'), ('M', 'D1'), ('Y', 'D2'), ('D1', 'D2')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y'), ('D1', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.6, size)
        U4 = np.random.uniform(-1, 0.2, size)
        U5 = np.random.binomial(1, 0.3, size)
        U6 = np.random.uniform(0.5, 1.5, size)
        U7 = np.random.binomial(1, 0.6, size)
        U8 = np.random.uniform(-1, 0.4, size)

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

        # D1 = f(C2, M, U7, U8)
        eps_d1 = np.random.normal(0, 1, size)
        D1 = 1 + C2 + M + U7 + U8 + eps_d1

        # Y = f(M, U5, U6, U7, U8)
        eps_y = np.random.normal(0, 1, size)
        Y = 1 + 1 * M + U5 + U6 + U7 + U8 + eps_y

        # D2 = f(Y, D1)
        p_d2 = expit(1 - Y + D1)
        D2 = np.random.binomial(1, p_d2, size)

        data = pd.DataFrame({'C1':C1, 'C2':C2, 'Z1':Z1, 'Z2':Z2, 'T':T, 'M':M, 'Y':Y, 'D1':D1, 'D2':D2})

        ace_truth = -0.5

        cmean = AverageCausalEffect(G, 'T', 'Y')
        ace_ipw, _ = cmean.bootstrap_ace(data, "ipw")
        ace_gformula, _ = cmean.bootstrap_ace(data, "gformula")
        ace_aipw, _ = cmean.bootstrap_ace(data, "aipw")
        ace_eif, _ = cmean.bootstrap_ace(data, "eif-aipw")

        self.assertTrue(abs(ace_ipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_gformula - ace_truth) < TOL)
        self.assertTrue(abs(ace_aipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_eif - ace_truth) < TOL)

    def test_a_fixability_not_mbshielded(self):
        np.random.seed(0)

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y', 'D1', 'D2']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y'), ('M', 'D1'), ('Y', 'D2'), ('D1', 'D2')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y'), ('D1', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.6, size)
        U4 = np.random.uniform(-1, 0.2, size)
        U5 = np.random.binomial(1, 0.3, size)
        U6 = np.random.uniform(0.5, 1.5, size)
        U7 = np.random.binomial(1, 0.6, size)
        U8 = np.random.uniform(-1, 0.4, size)

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

        # D1 = f(M, U7, U8)
        eps_d1 = np.random.normal(0, 1, size)
        D1 = 1 + M + U7 + U8 + eps_d1

        # Y = f(M, U5, U6, U7, U8)
        eps_y = np.random.normal(0, 1, size)
        Y = 1 + 1 * M + U5 + U6 + U7 + U8 + eps_y

        # D2 = f(Y, D1)
        p_d2 = expit(1 - Y + D1)
        D2 = np.random.binomial(1, p_d2, size)

        data = pd.DataFrame({'C1':C1, 'C2':C2, 'Z1':Z1, 'Z2':Z2, 'T':T, 'M':M, 'Y':Y, 'D1':D1, 'D2':D2})

        cmean = AverageCausalEffect(G, 'T', 'Y')
        self.assertFalse(cmean.is_mb_shielded)
        self.assertEqual(cmean.strategy, "a-fixable")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "eif-aipw")


    def test_p_fixability_mbshielded_1(self):
        vertices = ['C', 'T', 'M', 'L', 'Y']
        di_edges = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('M', 'Y'),
                    ('L', 'Y')]
        bi_edges = [('T', 'L'), ('T', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        cmean = AverageCausalEffect(G, 'T', 'Y')
        self.assertTrue(cmean.is_mb_shielded)
        self.assertEqual(cmean.strategy, "p-fixable")


    def test_p_fixability_mbshielded_2(self):
        vertices = ['C', 'T', 'M', 'L', 'Y']
        di_edges = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('T', 'Y'),
                    ('L', 'Y')]
        bi_edges = [('T', 'L'), ('M', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        cmean = AverageCausalEffect(G, 'T', 'Y')
        self.assertTrue(cmean.is_mb_shielded)
        self.assertEqual(cmean.strategy, "p-fixable")

    def test_p_fixability_compute_ace(self):
        np.random.seed(0)
        vertices = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
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

        data = pd.DataFrame({'C1':C1, 'C2':C2, 'Z1':Z1, 'Z2':Z2, 'T':T, 'M':M, 'L':L, 'Y':Y})

        ace_truth = -0.07

        cmean = AverageCausalEffect(G, 'T', 'Y')
        ace_pipw, _ = cmean.bootstrap_ace(data, "p-ipw")
        ace_dipw, _ = cmean.bootstrap_ace(data, "d-ipw")
        ace_apipw, _ = cmean.bootstrap_ace(data, "apipw")
        ace_eif, _ = cmean.bootstrap_ace(data, "eif-apipw")

        print(ace_pipw)
        print(ace_dipw)
        print(ace_apipw)
        print(ace_eif)
        self.assertTrue(abs(ace_pipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_dipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_apipw - ace_truth) < TOL)
        self.assertTrue(abs(ace_eif - ace_truth) < TOL)
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "ipw")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "gformula")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "aipw")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "eif-aipw")

    def test_p_fixability_not_mbshielded(self):
        np.random.seed(0)
        vertices = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'L'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
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
        p_t = expit(0.5 + 0.5 * C1 - 0.4 * U5 + 0.4 * U6)
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

        cmean = AverageCausalEffect(G, 'T', 'Y')
        self.assertFalse(cmean.is_mb_shielded)
        self.assertEqual(cmean.strategy, "p-fixable")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "eif-apipw")


    def test_front_door(self):
        np.random.seed(0)
        vertices = ['C', 'T', 'M', 'Y']
        di_edges = [('C', 'T'), ('C', 'M'), ('C', 'Y'), ('T', 'M'), ('M', 'Y')]
        bi_edges = [('T', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)

        C = np.random.normal(0, 1, size)

        # T = f(C, U1, U2)
        p_t = expit(0.5 + 0.5 * C - 0.4 * U1 + 0.4 * U2)
        T = np.random.binomial(1, p_t, size)

        # M = f(C, T)
        p_m = expit(-0.3 + 1.5 * T - 0.3 * C)
        M = np.random.binomial(1, p_m, size)

        # Y = f(C, M, U1, U2)
        eps_y = np.random.normal(0, 1, size)
        Y = 1 + C + 0.2*M + 0.4*U1 + 0.5*U2 + eps_y

        data = pd.DataFrame({'C': C, 'T': T, 'M': M, 'Y': Y, 'U1': U1, 'U2': U2})

        cmean = AverageCausalEffect(G, 'T', 'Y')

        # # Estimate ACE
        # formula = "Y ~ T + C + U1 + U2"
        # model = cmean._fit_continuous_glm(data, formula)
        # data_T1 = data.copy()
        # data_T1[cmean.treatment] = 1
        # data_T0 = data.copy()
        # data_T0[cmean.treatment] = 0
        # ace_truth = np.mean(model.predict(data_T1) - model.predict(data_T0))

        ace_truth = 0.064

        ace_pipw, _ = cmean.bootstrap_ace(data, "p-ipw")

        self.assertTrue(cmean.is_mb_shielded)
        self.assertEqual(cmean.strategy, "p-fixable")
        self.assertTrue(abs(ace_pipw - ace_truth) < TOL)


    def test_nested_fixability(self):
        vertices = ['C1',  'C2',  'T', 'M', 'Z', 'R1', 'R2', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'Y'), ('C2', 'T'), ('C2', 'Y'), ('R2', 'Y'), ('Z', 'T'),
                    ('T', 'R1'), ('T', 'Y'), ('R1', 'M'), ('M', 'Y')]
        bi_edges = [('Z', 'R2'), ('T', 'R2'), ('Z', 'R1'), ('C1', 'M'), ('C1', 'Y'), ('C2', 'M'), ('C2', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        size = 2000
        # U1, U2, U3, U4, U5, U6, U7, U8, U9, U10
        U1 = np.random.binomial(1, 0.4, size)
        U2 = np.random.uniform(0, 1.5, size)
        U3 = np.random.binomial(1, 0.3, size)
        U4 = np.random.uniform(0, 0.5, size)
        U5 = np.random.binomial(1, 0.4, size)
        U6 = np.random.uniform(0, 1.5, size)
        U7 = np.random.binomial(1, 0.3, size)
        U8 = np.random.uniform(0, 0.5, size)
        U9 = np.random.binomial(1, 0.3, size)
        U10 = np.random.uniform(0, 0.5, size)

        # R2 = f(U1, U2, U3, U4)
        p_r2 = expit(-0.2 + U1 - 0.8 * U2 + U3 + U4)
        R2 = np.random.binomial(1, p_r2, size)

        # C1 = f(U7, U8, U9, U10)
        eps_c1 = np.random.normal(0, 1, size)
        C1 = U7 - U8 + U9 + U10 + eps_c1

        # C2 = f(U7, U8)
        eps_c2 = np.random.normal(0, 1, size)
        C2 = U7 - U8 + U9 * U10 + eps_c2

        # Z = f(U1, U2, U5, U6)
        eps_z = np.random.normal(0, 1, size)
        Z = U1 - U2 - U5 - U6 + eps_z

        # T = f(C1, C2, Z, U3, U4)
        p_t1 = expit(0.8 - 0.5 * C1 + 0.5 * C2 + 0.3 * Z + 0.5 * U3 - 0.4 * U4)
        T = np.random.binomial(1, p_t1, size)

        # R1 = f(T, U5, U6)
        p_r1 = expit(0.2 + 0.7 * T - 0.6 * U5 - 0.6 * U6)
        R1 = np.random.binomial(1, p_r1, size)

        # M = f(R1, U7, U8)
        eps_m = np.random.normal(0, 1, size)
        M = 1 - R1 - U7 + U8 + eps_m

        # Y = f(M, T, U9, U10)
        eps_y = np.random.normal(0, 1, size)
        Y = 1 + R2 + M + 1 * T + C1 + C2 + U9 + U10 + eps_y

        data = pd.DataFrame({'C1': C1, 'C2': C2, 'T': T, 'M': M, 'Z': Z, 'R1': R1, 'R2': R2, 'Y': Y})

        cmean = AverageCausalEffect(G, 'T', 'Y')
        ace_nipw, _ = cmean.bootstrap_ace(data, "n-ipw")
        ace_anipw, _ = cmean.bootstrap_ace(data, "anipw")

        self.assertEqual(cmean.strategy, "nested-fixable")
        self.assertEqual(ace_nipw, 0)
        self.assertEqual(ace_anipw, 0)
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "p-ipw")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "d-ipw")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "apipw")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "eif-apipw")

    def test_bow_arc(self):
        vertices = ['C', 'T', 'Y']
        di_edges = [('C', 'T'), ('C', 'Y'), ('T', 'Y')]
        bi_edges = [('T', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)

        data = pd.DataFrame()
        cmean = AverageCausalEffect(G, 'T', 'Y')
        self.assertEqual(cmean.strategy, "Not ID")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "n-ipw")
        with self.assertRaises(RuntimeError):
            cmean.bootstrap_ace(data, "anipw")


if __name__ == '__main__':
    unittest.main()
