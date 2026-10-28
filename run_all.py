"""Reproduce the main results of "An acousto-optic colorimeter for color
psychophysics" from the calibration and measurements in this directory.
Runs on Linux, macOS and Windows with the interpreter it is started with:

    python run_all.py            # from anywhere; --jobs N to change the parallelism

The selection runs are single-threaded and independent, so they run JOBS at a
time (default 6; about 16 minutes on 8 cores). Results go to output/, logs to
output/logs/, and summary.py ends the run by printing the main results (also
saved to output/summary.txt).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "output" / "logs"
CENTERS = ["gray", "yellow", "green", "blue", "red"]
PY = sys.executable
ENV = {**os.environ,
       "PYTHONPATH": os.pathsep.join([".", "scripts"]),   # ao/ and scripts/common.py
       "PYTHONUTF8": "1"}                                 # UTF-8 output on every platform


def script(name, *args):
    return [PY, str(ROOT / "scripts" / name), *args]


def step(title, *cmd):
    """Run one command in the foreground, output to the console."""
    if title:
        print(f"== {title}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def batch(title, jobs, runs):
    """Run (log_name, command) pairs, at most ``jobs`` at a time, each logged
    to output/logs/<log_name>.log; stop at the first failure."""
    print(f"== {title}", flush=True)
    pending, active = list(runs), []
    while pending or active:
        while pending and len(active) < jobs:
            name, cmd = pending.pop(0)
            log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
            active.append((name, log, subprocess.Popen(cmd, cwd=ROOT, env=ENV,
                                                       stdout=log, stderr=subprocess.STDOUT)))
        name, log, proc = active[0]
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            active.append(active.pop(0))        # still running, look at the next one
            continue
        active.pop(0)
        log.close()
        if proc.returncode != 0:
            for _, l, p in active:
                p.kill()
                l.close()
            sys.exit(f"{name} failed (exit {proc.returncode}), see output/logs/{name}.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 6)),
                    help="selection runs in parallel (default 6)")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    step("Validation (2026-10-17 experiment)", *script("validation.py"))
    step("Reference display model", *script("display_model.py"))
    step("Observer data (downloaded from Zenodo on the first run, then cached in data/)",
         PY, "-c", "import common; common.asano_observers()")

    runs = []
    for c in CENTERS:
        runs.append((f"ring_step_selection_{c}_k4",
                     script("ring_step_selection.py", "--center", c, "--primaries", "4")))
        runs.append((f"standard_anchor_selection_{c}",
                     script("standard_anchor_selection.py", "--center", c)))
    for c in CENTERS:
        runs.append((f"ring_step_selection_{c}_k3",
                     script("ring_step_selection.py", "--center", c, "--primaries", "3")))
    runs.append(("k3_standard_anchor_best", script("k3_standard_anchor_best.py")))
    runs.append(("primary_selection_crossvalidate", script("primary_selection_crossvalidate.py")))
    batch("Primary selection: resolution and observer criteria, every center", args.jobs, runs)

    step("Adopted configurations", *script("resolution_sets.py"))
    step(None, *script("standard_anchor_gate_refine.py"))
    step("Main results", *script("summary.py"))


if __name__ == "__main__":
    main()
