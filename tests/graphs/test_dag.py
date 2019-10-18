import unittest

from ananke.graphs import DAG


class TestDAG(unittest.TestCase):
    def test_dag_can_be_initialised(self):
        di_edges = [("A", "B")]
        vertices = ["A", "B"]
        dag = DAG(vertices, di_edges=di_edges)
        self.assertTrue(DAG)
        # test empty DAG
        DAG()

    def test_intrinsic_sets_are_singletons(self):
        pass


if __name__ == '__main__':
    unittest.main()
