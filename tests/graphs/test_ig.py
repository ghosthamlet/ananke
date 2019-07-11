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

    def test_insert(self):
        vertices = ["A", "B", "C"]
        di_edges = []
        bi_edges = [("A", "B"), ("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)

        ig.insert(frozenset(["A", "C"]))

        self.assertEqual(set(ig.vertices),
                         set([frozenset(["A"]), frozenset(["B"]), frozenset(["C"]), frozenset(["A", "C"])]))
        #self.assertEqual(2, len(ig.di_edges))

        ig.insert(frozenset(["A", "B", "C"]))
        self.assertEqual(set(ig.vertices), set(
            [frozenset(["A", "B", "C"]), frozenset(["A"]), frozenset(["B"]), frozenset(["C"]), frozenset(["A", "C"])]))
        #self.assertEqual(4, len(ig.di_edges))

    def test_add_extra_biedges(self):
        vertices = ["A", "B", "C"]
        di_edges = []
        bi_edges = [("A", "B"), ("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)

        ig.insert(frozenset(["A", "C"]))
        ig.add_extra_biedges(frozenset(["A", "C"]))

        self.assertEqual(set(ig.vertices),
                         set([frozenset(["A"]), frozenset(["B"]), frozenset(["C"]), frozenset(["A", "C"])]))
        self.assertEqual(set(ig.district(frozenset(["A", "C"]))),
                         set([frozenset(["A"]), frozenset(["B"]), frozenset(["C"]), frozenset(["A", "C"])]))

    def test_merge_on_three_var_graph(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B")]
        bi_edges = [("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)

        ig.merge(frozenset(["B"]), frozenset(["C"]))
        self.assertEqual(len(ig.bi_edges), 1)
        #self.assertEqual(len(ig.di_edges), 3)
        ig.merge(frozenset(["A"]), frozenset(["C"]))
        ig.merge(frozenset(["B"]), frozenset(["A", "C"]))
        ig.merge(frozenset(["A", "B", "C"]), frozenset(["A", "C"]))
        #self.assertEqual(4, len(ig.di_edges))
        self.assertEqual(len(ig.bi_edges), 0)

    def test_get_intrinsic_sets_three_var_graph(self):
        vertices = ["A", "B", "C"]
        di_edges = [("A", "B")]
        bi_edges = [("B", "C"), ("A", "C")]

        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)

        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
        print(intrinsic_sets)
        self.assertEqual(
            {frozenset({'C'}), frozenset({'C', 'B', 'A'}), frozenset({'C', 'A'}), frozenset({'A'}), frozenset({'B'})},
            intrinsic_sets)

    def test_get_intrinsic_sets_complete_bidirected_three_var_graph(self):
        vertices = ["A", "B", "C"]
        di_edges = []
        bi_edges = [("A", "B"), ("B", "C"), ("A", "C")]
        admg = ADMG(vertices=vertices, bi_edges=bi_edges, di_edges=di_edges)
        ig = IG(admg=admg)
        intrinsic_sets = ig.get_intrinsic_sets()
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


if __name__ == '__main__':
    unittest.main()
