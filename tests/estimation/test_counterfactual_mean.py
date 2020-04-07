
import unittest
import numpy as np
from scipy.special import expit
import pandas as pd

from ananke.graphs import ADMG
from ananke.estimation import CounterfactualMean


class TestCounterfactualMean(unittest.TestCase):

    def test_a_fixability(self):
        np.random.seed(0)

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y', 'D1', 'D2']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'), ('C2', 'D1'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y'), ('M', 'D1'), ('Y', 'D2'), ('D1', 'D2')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y'), ('D1', 'Y')]
        order = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'Y', 'D1', 'D2']
        G = ADMG(vertices, di_edges, bi_edges)

        size = 10000
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

        cmean = CounterfactualMean(G, 'T', 'Y', order)
        cmean.bootstrap_ace(data, "ipw")
        cmean.bootstrap_ace(data, "gformula")
        #ace = cmean.ace(dat)
        #print(ace)

        # cmean_1 = cmean.estimate(dat, 1)
        # cmean_0 = cmean.estimate(dat, 0)
        # ace_ipw = cmean_1["ipw"] - cmean_0["ipw"]
        # ace_gformula = cmean_1["g-formula"] - cmean_0["g-formula"]
        # ace_gaipw = cmean_1["g-aipw"] - cmean_0["g-aipw"]
        # ace_eff = cmean_1["efficient-if"] - cmean_0["efficient-if"]
        #
        # print(ace_ipw)
        # print(ace_gformula)
        # print(ace_gaipw)
        # print(np.var(ace_eff))


    def test_p_fixability_1(self):
        vertices = ['C', 'T', 'M', 'L', 'Y']
        di_edges = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('M', 'Y'),
                    ('L', 'Y')]
        bi_edges = [('T', 'L'), ('T', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        cmean = CounterfactualMean(G, 'T', 'Y')


    def test_p_fixability_2(self):
        vertices = ['C', 'T', 'M', 'L', 'Y']
        di_edges = [('C', 'T'), ('C', 'M'), ('C', 'L'), ('C', 'Y'), ('T', 'M'), ('M', 'L'), ('T', 'Y'),
                    ('L', 'Y')]
        bi_edges = [('T', 'L'), ('M', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        cmean = CounterfactualMean(G, 'T', 'Y')


    def test_nested_fixability(self):
        vertices = ['C', 'T', 'M', 'Z', 'R1', 'R2', 'Y']
        di_edges = [('C', 'T'), ('C', 'Y'), ('R2', 'Y'), ('Z', 'T'), ('T', 'R1'), ('T', 'Y'), ('R1', 'M'),
                    ('M', 'Y')]
        bi_edges = [('Z', 'R2'), ('T', 'R2'), ('Z', 'R1'), ('C', 'M'), ('C', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        cmean = CounterfactualMean(G, 'T', 'Y')

if __name__ == '__main__':
    unittest.main()
