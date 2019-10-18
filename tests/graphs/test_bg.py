import unittest

from ananke.graphs import BG

class TestBG(unittest.TestCase):
    def test_bg_can_be_initialised(self):
        bi_edges = [("A", "B")]
        vertices = ["A", "B"]
        bg = BG(vertices, bi_edges=bi_edges)
        self.assertTrue(BG)
        # test empty BG
        BG()


if __name__ == '__main__':
    unittest.main()
