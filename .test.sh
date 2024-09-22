#!/usr/bin/env bash
set -euo pipefail

# Uninstall numpy if it's already installed
pip uninstall -y numpy

# Reinstall numpy
pip install numpy

python -m mypy .
python -m pytest -vv
