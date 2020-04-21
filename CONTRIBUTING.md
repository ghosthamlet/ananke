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

It is also necessary to install separate dev python packages:
```{bash}
pip3 install -r dev_requirements.txt
```

as well as separate system packages:

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

## Test Coverage

```{bash}
pytest --cov=ananke tests/  # to generate the base report

```

## Building docs

* To build docs, run `bash docs/run.sh`
* Add tutorial notebooks in `docs/notebooks`. These are automatically built into webpages.

