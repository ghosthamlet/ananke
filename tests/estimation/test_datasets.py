import unittest
from ananke.datasets import helpers


class TestCounterfactualMean(unittest.TestCase):

    def test_loaders(self):

        data = helpers.load_conditionally_ignorable_data()
        self.assertTrue(len(data) == 2000)
        self.assertTrue("ViralLoad" in data)

        data = helpers.load_afixable_data()
        self.assertTrue(len(data) == 2000)
        self.assertTrue("Education" in data)

        data = helpers.load_frontdoor_data()
        self.assertTrue(len(data) == 2000)
        self.assertTrue("ViralLoad2" in data)


if __name__ == '__main__':
    unittest.main()