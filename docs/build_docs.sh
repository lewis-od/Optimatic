#!/bin/bash

sphinx-apidoc -f -o source ../optimatic
make html
