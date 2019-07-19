#!/usr/bin/bash

sphinx-apidoc -f -o source/ ../ananke
#make html
python3 -m sphinx source/ build/
