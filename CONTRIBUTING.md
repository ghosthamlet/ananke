# Contributing

## Installation for Development

Create a virtual environment and install `ananke` in this environment for running.
```{bash}
git clone git@gitlab.com:causal/ananke.git

cd ananke

python3 -m venv env

source env/bin/activate # this activates your environment

pip3 install -e .
```

It is also necessary to install separate development python packages:
```{bash}
pip3 install -r dev_requirements.txt
```

as well as separate system packages mostly required for building the sphinx-notebook support:

```{bash}
# this will depend on your environment, e.g.
sudo apt install pandoc # ubuntu
sudo dnf install pandoc # fedora
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

## Before Pushing 
Consider using the following command to lint the code

`flake8 ananke/`


## Test Coverage

```{bash}
pytest --cov=ananke tests/  # to generate the base report

```

## Running tests through tox
Tox runs tests as if they were installed from the pypi repository. Run `tox` in the project root to run the pytest tests in a virtualenv with ananke installed as a non-editable package.

## Sphinx docs

## Building docs

* To build docs, run `bash docs/run.sh`
* Add tutorial notebooks in `docs/notebooks`. These are automatically built into webpages.
* Maintain a `references.bib` in `docs/source`. 

