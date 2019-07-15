import unittest

from ananke.graphs import IG, ADMG


class TestIG(unittest.TestCase):
    def test_init(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B")]
        bi_edges = [("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)
        print(ig.bi_edges)

    def test_get_intrinsic_sets_three_var_graph(self):
        vertices = ["A1", "B", "C"]
        di_edges = [("A1", "B")]
        bi_edges = [("B", "C"), ("A1", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
        print(intrinsic_sets)
        self.assertEqual(
            {frozenset({'C'}), frozenset({'C', 'B', 'A1'}), frozenset({'C', 'A1'}), frozenset({'A1'}), frozenset({'B'})},
            intrinsic_sets)

    def test_get_intrinsic_sets_complete_bidirected_three_var_graph(self):
        vertices = ["A", "B", "C"]
        di_edges = []
        bi_edges = [("A", "B"), ("B", "C"), ("A", "C")]
        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)
        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
        print(intrinsic_sets)
        self.assertEqual({frozenset({"A"}), frozenset({"B"}), frozenset({"C"}),
                          frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"B", "C"}),
                          frozenset({"A", "B", "C"})},
                         intrinsic_sets)

    def test_get_intrinsic_sets_four_var_graph(self):
        vertices = ["A", "B", "C", "D"]
        di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
        bi_edges = [("A", "C"), ("A", "D")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)
        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()

        self.assertEqual({frozenset({"A"}), frozenset({"B"}), frozenset({"C"}), frozenset({"D"}),
                          frozenset({"A", "C"}), frozenset({"A", "C", "D"})},
                         intrinsic_sets)

    def test_get_intrinsic_sets_five_var(self):
        vertices = ["A", "B", "C", "D", "Y"]
        di_edges = [("A", "B"), ("A", "C"), ("B", "C"),
                    ("C", "D"), ("D", "Y")]
        bi_edges = [("A", "C"), ("B", "D"), ("C", "Y")]
        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)
        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
        truth = {frozenset({'D'}), frozenset({'C', 'A'}), frozenset({'Y'}), frozenset({'D', 'B'}), frozenset({'B'}), frozenset({'A'}), frozenset({'C', 'Y', 'A'})}
        self.assertEqual(truth, intrinsic_sets)
        print(intrinsic_sets)
        print('-'*20)
        print(ig.get_heads_tails())

    def test_check_fixing_order(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B")]
        bi_edges = [("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
        print(intrinsic_sets)
        fixing_order = ig.iset_fixing_order_map
        print(fixing_order)
        self.assertEqual(
            ["C", "A"],
            fixing_order[frozenset(["B"])])

    def test_get_intrinsic_sets_in_cadmg(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B")]
        bi_edges = [("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)
        admg.fix(["B"])
        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
        self.assertEqual(
            {frozenset(["A"]), frozenset(["C"]), frozenset(["A", "C"])},
            intrinsic_sets)


if __name__ == '__main__':
    unittest.main()
