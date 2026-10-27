#!/bin/bash
# Reproduce the main results of "An acousto-optic colorimeter for color
# psychophysics" from the calibration and measurements in this directory.
# Run from anywhere:
#
#     python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
#     ./run_all.sh
#
# The selection runs are single-threaded and independent, so they run JOBS at a
# time (default 6; about 16 minutes on 8 cores). Results go to output/, logs to
# output/logs/, and summary.py ends the run by printing the main results
# (also saved to output/summary.txt).
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.:scripts
PY=${PYTHON:-python}
JOBS=${JOBS:-6}
CENTERS=(gray yellow green blue red)
mkdir -p output/logs

# run LOG CMD...: start CMD in the background, at most $JOBS at a time.
# (Polling rather than `wait -n`, which needs bash 4.3; macOS ships bash 3.2.)
pids=""
run() {
    local log=$1; shift
    while [ $(jobs -rp | wc -l) -ge "$JOBS" ]; do sleep 1; done
    "$@" > "output/logs/$log.log" 2>&1 &
    pids="$pids $!"
}
# wait_all: wait for every run; exit if one of them failed
wait_all() {
    local pid
    for pid in $pids; do
        wait "$pid" || { echo "a run failed, see output/logs/"; exit 1; }
    done
    pids=""
}

echo "== Validation (2026-10-17 campaign)"
$PY scripts/validation.py
echo "== Reference display model"
$PY scripts/display_model.py

echo "== Observer data (downloaded from Zenodo on the first run, then cached in data/)"
$PY -c "import common; common.asano_observers()"

echo "== Primary selection: resolution and observer criteria, every center"
for c in "${CENTERS[@]}"; do
    run "ring_step_selection_${c}_k4" $PY scripts/ring_step_selection.py --center "$c" --primaries 4
    run "standard_anchor_selection_${c}" $PY scripts/standard_anchor_selection.py --center "$c"
done
for c in "${CENTERS[@]}"; do
    run "ring_step_selection_${c}_k3" $PY scripts/ring_step_selection.py --center "$c" --primaries 3
done
run k3_standard_anchor_best $PY scripts/k3_standard_anchor_best.py
run primary_selection_crossvalidate $PY scripts/primary_selection_crossvalidate.py
wait_all

echo "== Adopted configurations"
$PY scripts/resolution_sets.py
$PY scripts/standard_anchor_gate_refine.py

echo "== Main results"
$PY scripts/summary.py
