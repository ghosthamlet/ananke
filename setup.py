from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='ananke-causal',
      version='0.1',
      description='Python',
      url='',
      author='Rohit Bhattacharya',
      author_email='rbhatta8@jhu.edu',
      install_requires=requirements,
      packages=['ananke'],
      zip_safe=False)
