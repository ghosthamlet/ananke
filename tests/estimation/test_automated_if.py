import unittest

from ananke.graphs import ADMG
from ananke.estimation import AutomatedIF


class TestADMG(unittest.TestCase):

    def test_conditionally_ignorable_model(self):

        vertices = ['A', 'C', 'Y']
        di_edges = [('C', 'A'), ('C', 'Y'), ('A', 'Y')]
        G = ADMG(vertices, di_edges)
        eif = AutomatedIF(G, 'A', 'Y')
        print("Top order:", eif.top_order_)
        print("Pretreatments:", eif.C, ", Mediators:", eif.M, ", Post district of A:", eif.L)

    def test_frontdoor_model(self):

        vertices = ['C', 'A', 'M', 'Y']
        di_edges = [('C', 'A'), ('C', 'M'), ('C', 'Y'), ('A', 'M'), ('M', 'Y')]
        bi_edges = [('A', 'Y')]
        G = ADMG(vertices, di_edges, bi_edges)
        eif = AutomatedIF(G, 'A', 'Y')
        print("Top order:", eif.top_order_)
        print("Pretreatments:", eif.C, ", Mediators:", eif.M, ", Post district of A:", eif.L)

    def test_complicated_graph(self):
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('B', 'C'), ('D', 'C'), ('C', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        eif = AutomatedIF(G, 'A', 'Y')
        print("Top order:", eif.top_order_)
        print("Pretreatments:", eif.C, ", Mediators:", eif.M, ", Post district of A:", eif.L)
        print(eif.if_)


if __name__ == '__main__':
    unittest.main()
