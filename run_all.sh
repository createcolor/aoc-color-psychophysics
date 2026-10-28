#!/bin/bash
# Kept for the old instructions; the runner is run_all.py (any platform).
cd "$(dirname "$0")" && exec "${PYTHON:-python}" run_all.py "$@"
