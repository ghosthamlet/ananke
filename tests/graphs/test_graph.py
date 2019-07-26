import unittest

from ananke.graphs import Graph


class TestGraph(unittest.TestCase):

    def test_directed_paths(self):

        vertices = ["A", "B", "C", "D", "Y"]
        di_edges = [("A", "B"), ("A", "D"), ("B", "C"), ("C", "Y"), ("B", "D"), ("D", "Y")]
        bi_edges = [("A", "C"), ("B", "Y"), ("B", "D")]
        G = Graph(vertices, di_edges, bi_edges)

        # test there are no directed paths from Y to A
        self.assertEqual([], G.directed_paths(["Y"], ["A"]))
        A_to_Y_paths = G.directed_paths(["A"], ["Y"])
        # TODO: How to assert equality of these paths
        true_paths = [[('A', 'D'), ('D', 'Y')], [('A', 'B'), ('B', 'D'), ('D', 'Y')], [('A', 'B'), ('B', 'C'), ('C', 'Y')]]
        print(A_to_Y_paths)

if __name__ == '__main__':
    unittest.main()