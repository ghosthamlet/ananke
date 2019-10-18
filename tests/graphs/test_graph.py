import unittest

from ananke.graphs import Graph


class TestGraph(unittest.TestCase):

    def test_graph_can_be_initialised(self):
        ud_edges = [("A", "B")]
        vertices = ["A", "B"]
        G = Graph(vertices, ud_edges=ud_edges)
        self.assertTrue(Graph)
        # test empty BG
        Graph()

    def test_simple_edge_operations(self):
        vertices = ["A", "a", "B", "C", "D", "Y"]
        di_edges = [("A", "a"), ("B", "C")]
        bi_edges = [("D", "Y"), ("A", "Y")]
        ud_edges = [("B", "C"), ("C", "D")]
        G = Graph(vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges)
        G.draw()

        # test neighbour relations
        self.assertEqual(G.neighbors(["C"]), set({"B", "D"}))

        # test that you can delete di edges
        G.delete_diedge("A", "a")
        self.assertTrue(("A", "a") not in G.di_edges)
        self.assertTrue("a" not in G.children(["A"]))
        self.assertTrue("A" not in G.parents(["a"]))

        # test that you can delete bi edges
        self.assertTrue(G.has_biedge("Y", "A"))
        G.delete_biedge("A", "Y")
        self.assertTrue(("A", "Y") not in G.bi_edges)
        self.assertFalse(G.has_biedge("Y", "A"))
        self.assertTrue("A" not in G.siblings(["Y"]))
        self.assertTrue("Y" not in G.siblings(["A"]))
        #self.assertRaises(KeyError, G.delete_biedge("A", "C"))
        #self.assertRaises(KeyError, G.delete_biedge("B", "D"))
        self.assertEqual(G.neighbors(["C"]), set({"B", "D"}))

    def test_directed_paths(self):

        vertices = ["A", "B", "C", "D", "Y"]
        di_edges = [("A", "B"), ("A", "D"), ("B", "C"), ("C", "Y"), ("B", "D"), ("D", "Y")]
        bi_edges = [("A", "C"), ("B", "Y"), ("B", "D")]
        G = Graph(vertices, di_edges, bi_edges)

        # test there are no directed paths from Y to A
        self.assertEqual([], G.directed_paths(["Y"], ["A"]))
        A_to_Y_paths = G.directed_paths(["A"], ["Y"])
        # TODO: How to assert equality of these paths
        true_paths = [[('A', 'D'), ('D', 'Y')], [('A', 'B'), ('B', 'D'), ('D', 'Y')],
                      [('A', 'B'), ('B', 'C'), ('C', 'Y')]]
        print(A_to_Y_paths)


if __name__ == '__main__':
    unittest.main()