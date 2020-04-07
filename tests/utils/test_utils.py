import unittest

from ananke.utils import powerset


class TestUtils(unittest.TestCase):

    def test_powerset(self):

        self.assertEqual(list(powerset([1, 2, 3])),
                         [(), (1, ), (2, ), (3, ), (1, 2), (1, 3), (2, 3), (1, 2, 3)])


if __name__ == '__main__':
    unittest.main()