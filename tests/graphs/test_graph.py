import unittest

from ananke.graphs import Graph


class TestGraph(unittest.TestCase):

    def test_graph_can_be_initialised(self):
        ud_edges = [('A', 'B')]
        vertices = ['A', 'B']
        G = Graph(vertices, ud_edges=ud_edges)
        self.assertTrue(Graph)
        # test empty BG
        Graph()

    def test_simple_edge_operations(self):
        vertices = ['A', 'a', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'a'), ('B', 'C')]
        bi_edges = [('D', 'Y'), ('A', 'Y')]
        ud_edges = [('B', 'C'), ('C', 'D')]
        G = Graph(vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges)

        # drawing tests (don't actually do anything)
        G.draw()
        G.draw(direction='LR')

        # test neighbour relations
        self.assertEqual(G.neighbors(['C']), set({'B', 'D'}))

        # test that you can delete di edges
        G.delete_diedge('A', 'a')
        self.assertTrue(('A', 'a') not in G.di_edges)
        self.assertTrue('a' not in G.children(['A']))
        self.assertTrue('A' not in G.parents(['a']))

        # test that you can delete bi edges
        self.assertTrue(G.has_biedge('Y', 'A'))
        G.delete_biedge('A', 'Y')
        self.assertTrue(('A', 'Y') not in G.bi_edges)
        self.assertFalse(G.has_biedge('Y', 'A'))
        self.assertTrue('A' not in G.siblings(['Y']))
        self.assertTrue('Y' not in G.siblings(['A']))

        # test that you can delete ud edges
        self.assertEqual(G.neighbors(['C']), set({'B', 'D'}))
        G.delete_udedge('D', 'C')
        self.assertTrue('D' not in G.neighbors(['C']))

    def test_directed_paths(self):

        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'Y'), ('B', 'D'), ('D', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = Graph(vertices, di_edges, bi_edges)

        # test there are no directed paths from Y to A
        self.assertEqual([], G.directed_paths(['Y'], ['A']))
        A_to_Y_paths = G.directed_paths(['A'], ['Y'])
        # TODO: How to assert equality of these paths
        true_paths = [[('A', 'D'), ('D', 'Y')], [('A', 'B'), ('B', 'D'), ('D', 'Y')],
                      [('A', 'B'), ('B', 'C'), ('C', 'Y')]]
        print(A_to_Y_paths)

    def test_subgraph(self):

        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('C', 'D')]
        ud_edges = [('C', 'D')]
        bi_edges = [('A', 'Y'), ('A', 'C'), ('A', 'B')]
        G = Graph(vertices, di_edges, bi_edges, ud_edges)
        G.vertices['A'].fixed = True

        Gsub = G.subgraph(['A', 'B', 'Y'])
        self.assertEqual(set(['A', 'B', 'Y']), set(Gsub.vertices))
        self.assertEqual(set([('A', 'B')]), Gsub.di_edges)
        self.assertEqual(set([]), Gsub.ud_edges)
        self.assertEqual(set([('A', 'Y'), ('A', 'B')]), Gsub.bi_edges)

    def test_topological_sort(self):

        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('B', 'C'), ('D', 'C'), ('C', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = Graph(vertices, di_edges, bi_edges)
        top_order = G.topological_sort()
        self.assertTrue(top_order == ['A', 'D', 'B', 'C', 'Y'] or
                        top_order == ['D', 'A', 'B', 'C', 'Y'] or
                        top_order == ['A', 'B', 'D', 'C', 'Y'])

    def test_pre(self):
        vertices = ['A', 'B', 'C', 'D', 'Y']
        di_edges = [('A', 'B'), ('B', 'C'), ('D', 'C'), ('C', 'Y')]
        bi_edges = [('A', 'C'), ('B', 'Y'), ('B', 'D')]
        G = Graph(vertices, di_edges, bi_edges)
        top_order = ['D', 'A', 'B', 'Y']
        self.assertEqual(G.pre(['D', 'A'], top_order), [])
        self.assertEqual(G.pre(['B', 'C'], top_order), ['D', 'A'])


if __name__ == '__main__':
    unittest.main()