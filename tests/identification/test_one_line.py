import unittest
from ananke.identification import OneLineID
from ananke.graphs import ADMG


class TestOneLine(unittest.TestCase):
    def test_id_graph(self):
        """
        Test that ADMG Y(a) is identified

        :return:
        """

        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        one_id = OneLineID(G, ['A'], ['Y'])
        self.assertTrue(one_id.id())
        self.assertEqual({'Y', 'C', 'D', 'B'}, one_id.ystar)

    def test_non_id_graph(self):
        """
        Test that Y(a,b) is not identified

        :return:
        """

        # non ID test
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        one_id = OneLineID(G, ['A', 'B'], ['Y'])
        self.assertFalse(one_id.id())


if __name__ == '__main__':
    unittest.main()
