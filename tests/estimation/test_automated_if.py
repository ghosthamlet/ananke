import unittest

from ananke.graphs import ADMG
from ananke.estimation import AutomatedIF


class TestADMG(unittest.TestCase):

    def test_conditionally_ignorable_model(self):

        vertices = ['A', 'C', 'Y']
        di_edges = [('C', 'A'), ('C', 'Y'), ('A', 'Y')]
        G = ADMG(vertices, di_edges)
        influence = AutomatedIF(G, 'A', 'Y')
        print(influence.beta_primal_)
        print(influence.beta_dual_)
        print(influence.nonparametric_if_)
        print(influence.eff_if_)
        self.assertEqual(influence.eff_if_, influence.nonparametric_if_)

    def test_mbshielded_afixable(self):

        vertices = ['Z1', 'Z2', 'C1', 'C2', 'T', 'M', 'Y']
        di_edges = [('C1', 'Z1'), ('C1', 'T'), ('C1', 'M'), ('C2', 'Z1'), ('C2', 'T'), ('C2', 'M'),
                    ('Z1', 'Z2'), ('Z2', 'T'), ('T', 'M'), ('M', 'Y')]
        bi_edges = [('Z1', 'T'), ('Z2', 'C1'), ('C2', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        influence = AutomatedIF(G, 'T', 'Y')
        print(influence.beta_primal_)
        print(influence.beta_dual_)
        print(influence.nonparametric_if_)
        print(influence.eff_if_)
        self.assertNotEqual(influence.eff_if_, influence.nonparametric_if_)

    def test_frontdoor_model(self):

        vertices = ['A', 'M', 'Y']
        di_edges = [('A', 'M'), ('M', 'Y')]
        bi_edges = [('A', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        influence = AutomatedIF(G, 'A', 'Y')
        print(influence.beta_primal_)
        print(influence.beta_dual_)
        print(influence.nonparametric_if_)
        print(influence.eff_if_)
        self.assertEqual(influence.eff_if_, influence.nonparametric_if_)

    def test_mbshielded_pfixable(self):

        vertices = ['C1', 'C2', 'Z1', 'Z2', 'T', 'M', 'L', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'L'), ('C2', 'T'), ('C2', 'M'), ('C2', 'L'), ('C2', 'Y'),
                    ('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('Z1', 'C1'), ('Z2', 'C2'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)
        influence = AutomatedIF(G, 'T', 'Y')
        print(influence.beta_primal_)
        print(influence.beta_dual_)
        print(influence.nonparametric_if_)
        print(influence.eff_if_)
        self.assertNotEqual(influence.eff_if_, influence.nonparametric_if_)

    def test_verma_afixable(self):
        vertices = ['T', 'M', 'L', 'Y']
        di_edges = [('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('M', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        influence = AutomatedIF(G, 'T', 'Y')
        print(influence.beta_primal_)
        print(influence.beta_dual_)
        print(influence.nonparametric_if_)
        print(influence.eff_if_)
        self.assertEqual(influence.eff_if_, "Cannot compute, graph is not mb-shielded.")

    def test_verma_pfixable(self):
        vertices = ['T', 'M', 'L', 'Y']
        di_edges = [('T', 'M'), ('M', 'L'), ('L', 'Y')]
        bi_edges = [('M', 'Y'), ('T', 'L')]
        G = ADMG(vertices, di_edges, bi_edges)
        influence = AutomatedIF(G, 'T', 'Y')
        print(influence.beta_primal_)
        print(influence.beta_dual_)
        print(influence.nonparametric_if_)
        print(influence.eff_if_)
        self.assertEqual(influence.eff_if_, "Cannot compute, graph is not mb-shielded.")

    def test_nested_fixable_and_notid(self):

        G_bowarc = ADMG(vertices=['T', 'Y'], di_edges=[('T', 'Y')], bi_edges=[('T', 'Y')])
        with self.assertRaises(RuntimeError):
            AutomatedIF(G_bowarc, 'T', 'Y')
        vertices = ['C1', 'C2', 'T', 'M', 'Z', 'R1', 'R2', 'Y']
        di_edges = [('C1', 'T'), ('C1', 'Y'), ('C2', 'T'), ('C2', 'Y'), ('R2', 'Y'), ('Z', 'T'),
                    ('T', 'R1'), ('T', 'Y'), ('R1', 'M'), ('M', 'Y')]
        bi_edges = [('Z', 'R2'), ('T', 'R2'), ('Z', 'R1'), ('C1', 'M'), ('C1', 'Y'), ('C2', 'M'), ('C2', 'Y')]
        G_nfixable = ADMG(vertices, di_edges, bi_edges)
        with self.assertRaises(NotImplementedError):
            AutomatedIF(G_nfixable, 'T', 'Y')


if __name__ == '__main__':
    unittest.main()
