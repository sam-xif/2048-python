#!/bin/sh
export PYTHONPATH=$PYTHONPATH:$PWD/src
python3 src/main.py $@
