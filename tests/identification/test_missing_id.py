import unittest
from ananke.identification import MissingFullID
from ananke.graphs import MissingADMG, ADMG


class TestMissingID(unittest.TestCase):

    def test_full_id(self):
        vertices = ['X_1', 'X_2', 'R_1', 'R_2']
        di_edges = [('X_1', 'X_2'), ('X_1', 'R_2'), ('R_2', 'R_1')]
        mdag = ADMG(vertices, di_edges)
        full_id = MissingFullID(mdag)
        self.assertTrue(full_id.id())

    def test_full_id_colluder(self):
        vertices = ['X_1', 'X_2', 'R_1', 'R_2']
        di_edges = [('X_1', 'R_2'), ('R_1', 'R_2')]
        mdag = ADMG(vertices, di_edges)
        full_id = MissingFullID(mdag)
        self.assertFalse(full_id.id())

    def test_full_id_cross_pattern(self):

        vertices = ['X_1', 'X_2', 'R_1', 'R_2']
        di_edges = [('X_1', 'R_2'), ('X_2', 'R_1'), ('R_2', 'R_1')]
        mdag = ADMG(vertices, di_edges)
        full_id = MissingFullID(mdag)
        self.assertFalse(full_id.id())

    def test_full_id_chain(self):

        vertices = ['X_1', 'X_2', 'X_3', 'R_1', 'R_2', 'R_3']
        di_edges = [('X_1', 'R_2'), ('X_2', 'R_3'), ('X_3', 'R_1'), ('R_3', 'R_2'), ('R_2', 'R_1')]
        mdag = ADMG(vertices, di_edges)
        full_id = MissingFullID(mdag)
        self.assertTrue(full_id.id())
        mdag.add_diedge('X_1', 'R_3')
        full_id = MissingFullID(mdag)
        self.assertTrue(full_id.id())


    def test_full_id_complicated(self):

        vertices = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5']
        di_edges = [('X_1', 'R_2'), ('X_2', 'R_3'), ('X_2', 'R_4'), ('X_2', 'R_5'), ('X_3', 'R_1'),
                    ('X_4', 'R_3'), ('X_4', 'R_5'), ('X_5', 'R_3'), ('X_5', 'R_4'),
                    ('R_3', 'R_2'), ('R_3', 'R_4'), ('R_3', 'R_5'), ('R_4', 'R_2'), ('R_5', 'R_2')]
        mdag = ADMG(vertices, di_edges)
        full_id = MissingFullID(mdag)
        self.assertTrue(full_id.id())
        mdag.add_diedge('R_2', 'R_1')
        full_id = MissingFullID(mdag)
        self.assertTrue(full_id.id())

    def test_full_not_ID(self):

        vertices = ['X_1', 'X_2', 'X_3', 'R_1', 'R_2', 'R_3', 'Xp_1', 'Xp_2', 'Xp_3']
        di_edges = [('X_1', 'X_2'), ('X_1', 'R_2'), ('X_1', 'R_3'),
                    ('X_2', 'R_3'), ('X_2', 'X_3'), ('X_3', 'R_1'), ('R_3', 'R_2'), ('R_2', 'R_1')]
        bi_edges = [('X_1', 'X_3'), ('X_2', 'R_3')]
        mdag = MissingADMG(vertices, di_edges, bi_edges)
        full_id = MissingFullID(mdag)
        self.assertTrue(full_id.id)
        mdag.add_biedge('R_1', 'R_3')
        full_id = MissingFullID(mdag)
        self.assertFalse(full_id.id())


if __name__ == '__main__':
    unittest.main()
