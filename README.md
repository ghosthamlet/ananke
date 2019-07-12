# Ananke

[Ananke](https://en.wikipedia.org/wiki/Ananke) named for the Greek
primordial goddess of necessity and causality, is a python package for
causal inference using the language of graphical models

## Installation and Development

Create a virtual environment and install `ananke` in this environment for running.
```{bash}
git clone git@gitlab.com:causal/ananke.git

cd ananke

python3 -m venv env

source env/bin/activate # this activates your environment

pip3 install -e .
```
Now you are ready to develop!

## Running Tests
To run tests locally (within the virtual environment):
```{bash}
python3 -m pytest tests/ # all tests

python3 -m pytest tests/graphs/test_admg.py # all tests in admg.py

python3 -m pytest tests/graphs/test_admg.py::TestADMG # a particular TestCase in admg.py

python3 -m pytest tests/graphs/test_admg.py::TestADMG::test_obtaining_districts # a particular TestCase method in admg.py
```

Continuous integration has been set up to run tests on pushes to the `master` branch.
