"""Best three-primary sets on the observer criterion: what the fourth primary
adds.

The k=3 counterpart of standard_anchor_selection.py: every three-primary set at
>= 20 nm separation with non-negative solver intensities is scored (three
primaries fix the coefficients, so there is no member to choose), ranked by
the training mean, and the ranking is walked down until a set renders the
center and its dE00 = 3 ring through the device path. That set's held-out
mean and worst are reported.

Usage:
    PYTHONPATH=.:scripts python scripts/k3_standard_anchor_best.py
Writes output/k3_standard_anchor_best.json.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from common import (CAL, CENTERS, MIN_SEPARATION_NM, OUT, RING_DE, Population, converter,
                    freqs_of, primaries_of, ring)


def main() -> None:
    conv = converter()
    pop = Population(conv)
    sc = pop.sc
    channels = conv._channel_assignment(3)
    results = {}
    for name in CENTERS:
        t = pop.center(name)
        rows = []
        for combo in combinations(primaries_of(sc), 3):
            wls = sorted((float(w) for w in combo), reverse=True)
            if np.any(np.diff(sorted(wls)) < MIN_SEPARATION_NM):
                continue
            try:
                inten = np.asarray(sc.waves_n_xyY_to_ao_intesities(wls, t["xyY"], channels=channels),
                                   dtype=float)
                if np.any(inten < 0):
                    continue
                stim = sum(np.asarray(sc._table_sd(wv, float(i), ch).values, float)
                           for wv, i, ch in zip(wls, inten, channels))
            except Exception:                       # noqa: BLE001  (not renderable)
                continue
            de = pop.de(stim, t)
            rows.append({"set": list(combo), "train_mean": float(de[pop.train].mean()),
                         **{f"test_{k}": v for k, v in pop.stats(de).items()}})
        rows.sort(key=lambda r: r["train_mean"])
        targets = [t["xyY"]] + ring(conv, t["xyY"], RING_DE)
        for rank, best in enumerate(rows, 1):
            freqs = freqs_of(conv, sorted(best["set"], reverse=True))
            if conv._feasible_for_targets([float(f) for f in freqs], targets):
                break
        else:
            print(f"--- {name}: no k=3 set passes the ring gate", flush=True)
            continue
        results[name] = dict(best, gate_rank=rank, n_renderable=len(rows))
        print(f"--- {name}: {len(rows)} renderable k=3 sets; rank-{rank} {best['set']} passes the ring "
              f"gate: held-out {best['test_mean']:.2f}/{best['test_worst']:.2f}", flush=True)
    (OUT / "k3_standard_anchor_best.json").write_text(json.dumps(
        {"calibration": CAL.name, "ring_de": RING_DE, "results": results}, indent=1))


if __name__ == "__main__":
    main()
