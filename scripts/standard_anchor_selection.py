"""Primaries for color difference: the observer criterion (paper Eq. 4) over every
four-primary set at one center.

Each set at >= 20 nm separation is solved for the display-derived center with
the converter's intensity solver; sets with a negative intensity are dropped.
The rendered spectrum is scored by the departure of every observer's percept
from the specified color (common.Population.de): the mean over the 75
training observers ranks the sets, the 76 held-out give the reported values.

Also written: the display's own departure on the same measure (the baseline)
and, via the rows, the spread of held-out means across sets.
standard_anchor_gate_refine.py takes the ranking from here.

Usage:
    PYTHONPATH=.:scripts python scripts/standard_anchor_selection.py --center yellow
Writes output/standard_anchor_selection/<center>.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations

import numpy as np

from common import CAL, CENTERS, MIN_SEPARATION_NM, OUT, Population, converter, primaries_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=CENTERS)
    args = ap.parse_args()
    conv = converter()
    pop = Population(conv)
    sc = pop.sc
    t = pop.center(args.center)
    channels = conv._channel_assignment(4)

    rows = []
    for n, combo in enumerate(combinations(primaries_of(sc), 4), 1):
        if n % 500 == 0:
            print(f"[{datetime.now():%H:%M:%S}] {args.center}: {n} combos, {len(rows)} renderable",
                  flush=True)
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
        except Exception:                           # noqa: BLE001  (not renderable)
            continue
        de = pop.de(stim, t)
        rows.append({"set": list(combo),
                     "train_mean": float(de[pop.train].mean()),
                     "test_mean": float(de[pop.test].mean()),
                     "test_worst": float(de[pop.test].max())})
    rows.sort(key=lambda r: r["train_mean"])

    display = pop.stats(pop.de_display(t))
    m = np.array([r["test_mean"] for r in rows])
    out = OUT / "standard_anchor_selection" / f"{args.center}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"center": args.center, "calibration": CAL.name,
                               "display_test": display, "rows": rows}, indent=1))
    print(f"--- {args.center}: {len(rows)} renderable; best by train {rows[0]['set']} "
          f"(test {rows[0]['test_mean']:.2f}/{rows[0]['test_worst']:.2f}); "
          f"held-out means {m.min():.2f}..{m.max():.2f} (x{m.max() / m.min():.1f}, "
          f"SD {m.std() / m.mean():.0%} of mean); display {display['mean']:.2f}/{display['worst']:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
