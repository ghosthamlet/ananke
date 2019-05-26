import unittest
from ananke.graphs import SG


class TestSG(unittest.TestCase):
    def test_non_segregated_graph_raises_error(self):
        vertices = ['A', 'B', 'C']
        bi_edges = [('A', 'B')]
        ud_edges = [('B', 'C')]
        with self.assertRaises(AssertionError):
            G = SG(vertices, bi_edges=bi_edges, ud_edges=ud_edges)

    def test_directed_cycle_graph_raises_error(self):
        # directed cycle graph
        vertices = ['A', 'B', 'C']
        bi_edges = [('A', 'B')]
        di_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
        with self.assertRaises(AssertionError):
            G = SG(vertices, bi_edges=bi_edges, di_edges=di_edges)

    def test_partially_directed_graph_raises_error(self):
        # partially directed cycle graph
        vertices = ['A', 'B', 'C']
        bi_edges = []
        ud_edges = [('A', 'B'), ('B', 'C')]
        di_edges = [('C', 'A')]
        with self.assertRaises(AssertionError):
            G = SG(vertices, bi_edges=bi_edges, di_edges=di_edges, ud_edges=ud_edges)

    def test_sg_does_not_raise_error(self):
        """
        This properly formed SG should not raise an AssertionError

        :return:
        """

        try:
            vertices = ['A', 'B', 'C']
            bi_edges = []
            ud_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
            G = SG(vertices, bi_edges=bi_edges, ud_edges=ud_edges)
        except AssertionError:
            self.fail("Correctly formed SG raised AssertionError")

    def test_that_districts_correct(self):

        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = SG(vertices, di_edges, bi_edges)
        self.assertCountEqual(([{'A', 'C'}, {'D', 'Y', 'B'}]), (G.districts()))
        self.assertEqual({'A', 'C'}, G.district('A'))
        #print(G.districts())
        #print(G.district('A'))

    def test_genealogical_sets_correct(self):
        vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
        di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
                    ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
        bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
        G = SG(vertices, di_edges, bi_edges)
        self.assertCountEqual((G.districts()), [{'X2', 'X1', 'U'}, {'A1'}, {'A2'}, {'Y2', 'Y1'}])
        #print(G.districts())
        self.assertEqual(G.district('X2'), {'X2', 'X1', 'U'})
        #print(G.district('X2'))
        self.assertEqual(G.ancestors('A2'), {'X2', 'A2', 'U'})
        #print(G.ancestors('A2'))
        self.assertEqual(G.ancestors('A1'), {'X1', 'A1', 'U'})
        #print(G.ancestors('A1'))
        self.assertEqual(G.ancestors(['A1', 'A2']), {'X2', 'A1', 'X1', 'U', 'A2'})
        #print(G.ancestors(['A1', 'A2']))
        self.assertEqual(G.descendants(['A1', 'A2']), {'Y2', 'Y1', 'A1', 'A2'})
        #print(G.descendants(['A1', 'A2']))

if __name__ == '__main__':
    unittest.main()
