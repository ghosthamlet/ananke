import unittest

from ananke.graphs import ADMG


class TestADMG(unittest.TestCase):
    def test_obtaining_districts(self):
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        #print(G.districts())
        self.assertCountEqual(([{'C', 'A'}, {'Y', 'B', 'D'}]), (G.districts))
        #print(G.district('A'))
        self.assertEqual({'C', 'A'}, G.district('A'))

    def test_obtaining_districts_large(self):
        vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
        di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
                    ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
        bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
        G = ADMG(vertices, di_edges, bi_edges)
        #print(G.districts())
        self.assertCountEqual(([{'X1', 'X2', 'U'}, {'A1'}, {'A2'}, {'Y1', 'Y2'}]), (G.districts))
        #print(G.district('X2'))
        self.assertEqual(G.district('X2'), {'X1', 'X2', 'U'})

        paths = list(G.m_connecting_paths('X1', 'Y2'))
        true_paths = [[('X1', 'A1', '->'), ('A1', 'Y2', '->')], [('X1', 'X2', '<->'), ('X2', 'Y2', '->')],
                      [('X1', 'U', '<->'), ('U', 'Y2', '->')],
                      [('X1', 'X2', '<->'), ('X2', 'A2', '->'), ('A2', 'Y2', '->')],
                      [('X1', 'U', '<->'), ('U', 'A1', '->'), ('A1', 'Y2', '->')],
                      [('X1', 'U', '<->'), ('U', 'A2', '->'), ('A2', 'Y2', '->')]]

        sorted_true_paths = [sorted(path) for path in sorted(true_paths)]
        self.assertEqual(sorted_true_paths,
                         [sorted(path) for path in sorted(paths)])

    def test_reachable_closure(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B"), ("C", "B"), ("C", "A")]
        bi_edges = [("A", "B")]
        G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
        cl, _, _ = G.reachable_closure(["B"])
        self.assertEqual({"A", "B"}, set(cl))


    def test_marg_projection(self):

        vertices = ["A", "B", "C"]
        di_edges = [("A", "B"), ("B", "C")]
        bi_edges = [("B", "C")]
        G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
        marg = G.maximal_arid_projection()
        #self.assertEqual(set([("A", "B"), ("B", "C"), ("A", "C")]), marg.di_edges)
        #self.assertEqual(set([]), marg.bi_edges)

if __name__ == '__main__':
    unittest.main()
