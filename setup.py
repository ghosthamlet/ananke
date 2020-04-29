from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='ananke-causal',
      version='0.1.2',
      description='Python',
      url='',
      author='Rohit Bhattacharya',
      author_email='rbhatta8@jhu.edu',
      install_requires=requirements,
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      zip_safe=False)
