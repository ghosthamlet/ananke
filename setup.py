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
      version='0.1.5',
      description='Causal inference with graphical models',
      long_description=('Ananke, named for the Greek primordial goddess of necessity and causality,' +
                        'is a python package for causal inference using the language of graphical models.\n\n' +
                        'Documentation: https://ananke.readthedocs.io/en/latest/index.html'),
      url='https://gitlab.com/causal/ananke',
      author='See AUTHORS',
      author_email='rbhatta8@jhu.edu',
      install_requires=requirements,
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_data={'ananke.datasets': ['simulated/*.csv']},
      zip_safe=False)
