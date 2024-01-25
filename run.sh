#!/bin/sh

python3 -m venv /tmp/lobster_tools_venv
source /tmp/lobster_tools_venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install .
python3 -m tests.test_sim