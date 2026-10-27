"""Transfer: what the observer criterion costs when the selection is made on a
population other than the one it serves.

The selection is fitted on observers other than the 151 it is reported on:
the CIE 2015 10 deg observer, the Stiles & Burch 1959 10 deg and 1955 2 deg
observers, and the Asano model swept over age 20-70 (individual variation at
zero). Every four-primary set with its coefficient family sampled at N_TAU
points is a candidate (linear model of the calibrated spectra at their top
level); each is scored by the mean departure from the specified color in each
observer's own CIELAB. Per center:

  chosen by test   the best candidate on the 151 (the in-sample optimum)
  chosen by train  the best candidate on the training observers, scored on the 151
  gap              the second minus the first: the cost of the transfer

Usage:
    PYTHONPATH=.:scripts python scripts/primary_selection_crossvalidate.py
Writes output/primary_selection_crossvalidate.json.
"""
from __future__ import annotations

import json

# common before colour: it silences colour's matplotlib warning
from common import (BATCH, CAL, OUT, adapting_white, candidates, converter, labs_of,
                    load_observers, observer_matrices, parse_monitor_spectra, primaries_of)

import numpy as np
from colour.difference import delta_E_CIE2000

AGE_SWEEP = list(range(20, 71, 5))
TRAIN_GROUPS = {"standard", "age_model"}


def main() -> None:
    conv = converter()
    sc = conv._spectral_converter
    grid = np.asarray(sc.grid, dtype=float)
    dlam = float((grid[-1] - grid[0]) / (len(grid) - 1))
    colors = parse_monitor_spectra()
    observers = load_observers(age_model_ages=AGE_SWEEP)
    reference = observers[0]
    ref_cmf = observer_matrices([reference], grid)[0]
    cmfs_test = observer_matrices([o for o in observers if o["group"] == "asano"], grid)
    train = [o for o in observers if o["group"] in TRAIN_GROUPS]
    cmfs_train = observer_matrices(train, grid)
    print(f"train {len(train)} observers: {', '.join(o['label'] for o in train)}", flush=True)

    white_test = adapting_white(colors, sc, reference, grid, dlam, ref_cmf, cmfs_test)
    white_train = adapting_white(colors, sc, reference, grid, dlam, ref_cmf, cmfs_train)
    white_std = adapting_white(colors, sc, reference, grid, dlam, ref_cmf, ref_cmf[None])

    prim_wl = primaries_of(sc)
    basis = np.array([np.clip(np.asarray(sc.spectra_table[k][1][-1], dtype=float), 0, None)
                      for k in prim_wl])
    basis = basis / ((basis @ ref_cmf[:, 1] * dlam)[:, None])
    A_ref = (basis @ ref_cmf) * dlam

    def mean_de(stims, cmfs, white, lab_t):
        lab = labs_of(np.einsum("bw,owc->boc", stims, cmfs) * dlam, white)
        return delta_E_CIE2000(np.broadcast_to(lab_t, lab.shape), lab).mean(axis=1)

    results = {}
    for c in colors:
        name = c["name"].strip()
        mon = np.interp(grid, c["wavelengths"], c["spectrum"], left=0.0, right=0.0)
        target = (mon @ ref_cmf) * dlam
        lab_test = labs_of(np.tile(target, (cmfs_test.shape[0], 1))[None],
                           np.tile(white_std, (cmfs_test.shape[0], 1)))[0]
        lab_train = labs_of(np.tile(target, (cmfs_train.shape[0], 1))[None],
                            np.tile(white_std, (cmfs_train.shape[0], 1)))[0]
        rows, buf, meta = [], [], []

        def flush():
            if buf:
                s = np.array(buf)
                rows.extend(zip(mean_de(s, cmfs_train, white_train, lab_train),
                                mean_de(s, cmfs_test, white_test, lab_test), meta))
                buf.clear()
                meta.clear()

        for combo, coef in candidates(A_ref, target, 4):
            buf.append(coef @ basis[list(combo)])
            meta.append([prim_wl[i] for i in combo])
            if len(buf) >= BATCH:
                flush()
        flush()
        by_train = min(rows, key=lambda r: r[0])
        by_test = min(rows, key=lambda r: r[1])
        results[name] = {"n_candidates": len(rows),
                         "chosen_by_test": {"set": by_test[2], "test_dE": float(by_test[1])},
                         "chosen_by_train": {"set": by_train[2], "test_dE": float(by_train[1])},
                         "gap": float(by_train[1] - by_test[1])}
        print(f"{name:12s} {len(rows):6d} candidates | by test {by_test[1]:.3f} {by_test[2]} | "
              f"by train {by_train[1]:.3f} {by_train[2]} | gap {by_train[1] - by_test[1]:+.3f}",
              flush=True)
    (OUT / "primary_selection_crossvalidate.json").write_text(json.dumps(
        {"calibration": CAL.name, "train_observers": [o["label"] for o in train],
         "results": results}, indent=1))


if __name__ == "__main__":
    main()
