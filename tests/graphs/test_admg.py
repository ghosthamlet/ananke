import unittest

from ananke.graphs import ADMG


class TestADMG(unittest.TestCase):

    def test_obtaining_districts(self):
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        self.assertCountEqual(([{'C', 'A'}, {'Y', 'B', 'D'}]), (G.districts))
        self.assertEqual({'C', 'A'}, G.district('A'))

    def test_obtaining_districts_large(self):
        vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
        di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
                    ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
        bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
        G = ADMG(vertices, di_edges, bi_edges)
        self.assertCountEqual(([{'X1', 'X2', 'U'}, {'A1'}, {'A2'}, {'Y1', 'Y2'}]), (G.districts))
        self.assertEqual(G.district('X2'), {'X1', 'X2', 'U'})

    def test_fixed(self):
        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G = ADMG(vertices, di_edges, bi_edges)
        G.fix(["X_1"])
        self.assertEqual(G.fixed, ["X_1"])

    def test_is_subgraph(self):
        vertices = ["X_1",  "W"]
        di_edges = [("X_1", "W"), ]
        bi_edges = [("X_1", "W")]
        G = ADMG(vertices, di_edges, bi_edges)
        G.fix(["X_1"])
        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G1 = ADMG(vertices, di_edges, bi_edges)
        self.assertTrue(G.is_subgraph(G1))

    def test_is_ancestral_subgraph(self):
        vertices = ["X_1",  "W"]
        di_edges = [("X_1", "W") ]
        bi_edges = [("X_1", "W")]
        G = ADMG(vertices, di_edges, bi_edges)
        G.fix(["X_1"])
        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G1 = ADMG(vertices, di_edges, bi_edges)
        G1.fix(["X_1"])
        self.assertTrue(G.is_ancestral_subgraph(G1))

        vertices = ["X_2", "W", "Y"]
        di_edges = [("W", "Y"), ("X_2", "Y")]
        bi_edges = [ ("X_2", "Y")]
        G = ADMG(vertices, di_edges, bi_edges)
        vertices = ["X_1", "X_2", "W", "Y"]
        di_edges = [("X_1", "W"), ("W", "Y"), ("X_2", "Y")]
        bi_edges = [("X_1", "W"), ("X_2", "Y"), ("X_1", "X_2")]
        G1 = ADMG(vertices, di_edges, bi_edges)
        G1.fix(["X_1"])
        self.assertFalse(G.is_ancestral_subgraph(G1))

    def test_subgraph(self):

        vertices = ["A", "B", "C", "D", "Y"]
        di_edges = [("A", "B"), ("C", "D")]
        bi_edges = [("A", "Y"), ("A", "C"), ("A", "B")]
        G = ADMG(vertices, di_edges, bi_edges)
        G.vertices["A"].fixed = True

        Gsub = G.subgraph(["A", "B", "Y"])
        self.assertEqual(set(["A", "B", "Y"]), set(Gsub.vertices))
        self.assertEqual(set([("A", "B")]), Gsub.di_edges)
        self.assertEqual(set([("A", "Y"), ("A", "B")]), Gsub.bi_edges)

    # def test_m_connecting_paths(self):
    #
    #     vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
    #     di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
    #                 ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
    #     bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
    #     G = ADMG(vertices, di_edges, bi_edges)
    #     paths = list(G._m_connecting_paths('X1', 'Y2'))
    #     true_paths = [[('X1', 'A1', '->'), ('A1', 'Y2', '->')], [('X1', 'X2', '<->'), ('X2', 'Y2', '->')],
    #                   [('X1', 'U', '<->'), ('U', 'Y2', '->')],
    #                   [('X1', 'X2', '<->'), ('X2', 'A2', '->'), ('A2', 'Y2', '->')],
    #                   [('X1', 'U', '<->'), ('U', 'A1', '->'), ('A1', 'Y2', '->')],
    #                   [('X1', 'U', '<->'), ('U', 'A2', '->'), ('A2', 'Y2', '->')]]
    #
    #     sorted_true_paths = [sorted(path) for path in sorted(true_paths)]
    #     self.assertEqual(sorted_true_paths,
    #                      [sorted(path) for path in sorted(paths)])
    #
    # def test_m_separation(self):
    #
    #     vertices = ['X1', 'U', 'X2', 'A1', 'A2', 'Y1', 'Y2']
    #     di_edges = [('X1', 'A1'), ('X1', 'Y1'), ('A1', 'Y1'), ('X2', 'A2'), ('X2', 'Y2'), ('A2', 'Y2'),
    #                 ('U', 'A1'), ('U', 'Y1'), ('U', 'A2'), ('U', 'Y2'), ('A2', 'Y1'), ('A1', 'Y2')]
    #     bi_edges = [('X1', 'U'), ('U', 'X2'), ('X1', 'X2'), ('Y1', 'Y2')]
    #     G = ADMG(vertices, di_edges, bi_edges)
    #     print(G._m_separation('X1', 'A2', ['X2']))
    #     print(list(G._m_connecting_paths('X1', 'A2', ['X2'])))


    def test_fixing(self):
        G = ADMG(vertices=["A", "B", "C"], di_edges=[("A", "B"), ("B", "C")], bi_edges=[("B", "C")])
        G.fix(["C"])
        self.assertTrue(("B", "C") not in G.di_edges)
        self.assertTrue(("B", "C") not in G.bi_edges)
        self.assertTrue(G.vertices["C"].fixed)

    def test_reachable_closure(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B"), ("C", "B"), ("C", "A")]
        bi_edges = [("A", "B")]
        G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
        cl, _, _ = G.reachable_closure(["B"])
        self.assertEqual({"A", "B"}, set(cl))

    def test_marg_dag_projection(self):

        vertices = ["A", "B", "C"]
        di_edges = [("A", "B"), ("B", "C")]
        bi_edges = [("B", "C")]
        G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
        marg = G.maximal_arid_projection()
        self.assertEqual(set([("A", "B"), ("B", "C"), ("A", "C")]), marg.di_edges)
        self.assertEqual(set([]), marg.bi_edges)

    def test_marg_biedge_projection(self):

        vertices = ["A", "B", "C"]
        di_edges = [("B", "A"), ("B", "C")]
        bi_edges = [("A", "B"), ("B", "C")]
        G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
        marg = G.maximal_arid_projection()
        self.assertEqual(set([("B", "A"), ("B", "C")]), marg.di_edges)
        self.assertEqual(set([("A", "C")]), marg.bi_edges)

    def test_marg_4var_graph(self):

        vertices = ["A", "B", "C", "D"]
        di_edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]
        bi_edges = [("A", "C"), ("B", "D")]
        G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        marg = G.maximal_arid_projection()
        self.assertEqual(set([("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]), marg.di_edges)
        self.assertEqual(set([("B", "D")]), marg.bi_edges)

    def test_markov_pillow(self):
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('B', 'C'), ('D', 'C'), ('C', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        top_order = ['D', 'A', 'B', 'C', 'Y']
        self.assertEqual(G.markov_pillow(['A', 'D'], top_order), set())
        self.assertEqual(G.markov_pillow(['C'], top_order), set(['A', 'B', 'D']))

    def test_markov_blanket(self):
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('B', 'C'), ('D', 'C'), ('C', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = ADMG(vertices, di_edges, bi_edges)
        self.assertEqual(G.markov_blanket(['A', 'D']), set(['C', 'B', 'Y']))
        self.assertEqual(G.markov_blanket(['C']), set(['A', 'B', 'D']))

    def test_nps(self):
        vertices = ['Treatment', 'M', 'L', 'Confounders', 'Outcome']
        di_edges = [('Confounders', 'M'), ('Confounders', 'L'),
                    ('Treatment', 'M'), ('Treatment', 'Outcome'), ('Treatment', 'L'),
                    ('M', 'L'), ('L', 'Outcome')]
        bi_edges_G1 = [('Treatment', 'Confounders'), ('M', 'Outcome'), ('L', 'Outcome')]
        bi_edges_G2 = [('Treatment', 'Confounders'), ('M', 'Outcome')]

        G1 = ADMG(vertices, di_edges, bi_edges_G1)
        G2 = ADMG(vertices, di_edges, bi_edges_G2)

        self.assertTrue(G1.nonparametric_saturated())
        self.assertFalse(G2.nonparametric_saturated())

        vertices = ['A', 'B', 'C']
        di_edges = [('B', 'A'), ('B', 'C')]
        bi_edges = [('A', 'B'), ('C', 'B')]
        G3 = ADMG(vertices, di_edges, bi_edges)

        self.assertTrue(G3.nonparametric_saturated())


if __name__ == '__main__':
    unittest.main()
