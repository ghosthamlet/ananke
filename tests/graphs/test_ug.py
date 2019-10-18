import unittest

from ananke.graphs import UG


class TestUG(unittest.TestCase):
    def test_ug_can_be_initialised(self):
        ud_edges = [("A", "B")]
        vertices = ["A", "B"]
        ug = UG(vertices, ud_edges=ud_edges)
        self.assertTrue(UG)
        # test empty BG
        UG()


if __name__ == '__main__':
    unittest.main()
