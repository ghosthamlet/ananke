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
      version='0.1.3',
      description='Python',
      url='',
      author='Rohit Bhattacharya',
      author_email='rbhatta8@jhu.edu',
      install_requires=requirements,
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_data={'ananke.datasets': ['simulated/*.csv']},
      zip_safe=False)
