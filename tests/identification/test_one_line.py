import unittest
from ananke.identification import OneLineID
from ananke.graphs import ADMG
from ananke import identification


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

    def test_two_var_id_graph(self):

        vertices = ["A", "D", "C", "Y"]
        di_edges = [('D', 'Y'), ('D', 'A'), ('A', 'Y'), ('C', 'A'), ('C', 'Y')]
        bi_edges = []
        G = ADMG(vertices, di_edges, bi_edges)
        one_id = OneLineID(G, ['A'], ['Y', 'D'])
        one_id.draw_swig(direction='LR').render()
        #G.draw(direction='LR').render()
        print(one_id.ystar)
        print(one_id.id())
        print(one_id.functional())
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

    def test_get_intrinsic_sets(self):

        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G = ADMG(vertices, di_edges, bi_edges)

        result = identification.get_intrinsic_sets(G)
        truth = {frozenset({"W", "X_1"}),
                 frozenset({"X_2"}),
                 frozenset({"X_1"}),
                 frozenset({"X_1", "X_2"}),
                 frozenset({"X_1", "X_2", "W"}),
                 frozenset({"X_1", "X_2", "W", "Y"})}


        self.assertEqual(truth, result)

class TestOneLineGZID(unittest.TestCase):
    def test_fix(self):
        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G = ADMG(vertices, di_edges, bi_edges)
        G.fix(["X_1"])
        S = identification.get_intrinsic_sets(G)

        print(S)


    def test_is_id(self):
        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G = ADMG(vertices, di_edges, bi_edges)
        interventions = ["X_1", "X_2"]
        outcomes = ["Y"]
        ol = identification.OneLineGZID(G, interventions, outcomes)
        status = ol._is_id()

        self.assertFalse(status)

        second = ol._is_id([{"X_1"}, {"X_2"}])
        self.assertTrue(second)


if __name__ == '__main__':
    unittest.main()
