from setuptools import setup, find_packages

requirements = [
        "networkx",
        "graphviz",
        "autograd",
        "scipy",
        "numpy",
        "pandas",
        "statsmodels"
]

setup(name='ananke-causal',
      version='0.1.4',
      description='Ananke, named for the Greek primordial goddess of necessity and causality, is a python package for causal inference using the language of graphical models.',
      url='https://gitlab.com/causal/ananke',
      author='Rohit Bhattacharya',
      author_email='rbhatta8@jhu.edu',
      install_requires=requirements,
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_data={'ananke.datasets': ['simulated/*.csv']},
      zip_safe=False)
