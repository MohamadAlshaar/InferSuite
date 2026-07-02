#!/usr/bin/env bash
# One-time box setup for the BCB co-located run: compile evblock, build BCB venv + task libs,
# verify the hard dataset loads. Run with nohup (installs take a few minutes).
set -uo pipefail
cd "$HOME/bcb"
echo "=== compile evblock.so ==="
gcc -O2 -fPIC -shared -o cudasync/evblock.so cudasync/evblock.c -ldl && echo "evblock.so built" || echo "evblock BUILD FAILED"
echo "=== bcb venv ==="
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
echo "=== bigcodebench (dataset loader) ==="
./.venv/bin/pip install -q bigcodebench==0.2.5
echo "=== core task libs (subset for non-network hard tasks) ==="
./.venv/bin/pip install -q numpy pandas scipy scikit-learn matplotlib sympy networkx \
  pillow openpyxl python-dateutil pytz statsmodels seaborn requests lxml beautifulsoup4 \
  nltk pytest prettytable texttable natsort wordninja xmltodict
echo "=== verify dataset load ==="
./.venv/bin/python -c "from bigcodebench.data import get_bigcodebench; d=get_bigcodebench(subset='hard'); print('HARD_TASKS', len(d))"
echo "SETUP_DONE"
