"""Adopt the resolution-selected configuration per center and score it on the
observer criterion, for the comparison of the two criteria (paper: Tables 2-3).

From the ring_step_selection outputs: the set with the lowest ring-scored J
over three and four primaries, among the sets that light every channel
(drive >= MIN_DRIVE). The condition only bites at gray, where the J-best set
drives 500 nm at 0.7 and the runner-up (573/510/473/450, J 0.228 against
0.2276) is adopted.

The observer score re-solves the adopted set at the display-derived center
through the device path, rounds the powers to 0.1, anchors luminance and
reports the held-out mean and worst departure (conventions of common.py).

Usage:
    PYTHONPATH=.:scripts python scripts/resolution_sets.py
Writes output/resolution_sets.json.
"""
from __future__ import annotations

import json

import numpy as np

from common import CENTERS, OUT, Population, converter, freqs_of

MIN_DRIVE = 1.0


def main() -> None:
    conv = converter()
    pop = Population(conv)
    out = {}
    for name in CENTERS:
        runs = {k: json.loads((OUT / "ring_step_selection" /
                               f"{name}_k{k}.json").read_text()) for k in (3, 4)}
        best = {k: min((r for r in run["rows"] if "J_ring" in r and min(r["powers"]) >= MIN_DRIVE),
                       key=lambda r: r["J_ring"]) for k, run in runs.items()}
        k = min(best, key=lambda k: best[k]["J_ring"])
        r, ch = best[k], runs[k]["channels"]

        t = pop.center(name)
        freqs = [float(f) for f in freqs_of(conv, r["set"])]
        P, _ = conv.xyY_to_P_f(list(t["xyY"]), frequencies=freqs)
        P = np.round(np.asarray([P[c - 1] for c in ch], float), 1)
        stim = np.asarray(conv.P_f_to_sd(P, np.asarray(freqs), channels=ch).values, float)
        obs = pop.stats(pop.de(stim, t))
        out[name] = {"set": r["set"], "channels": ch, "powers": r["powers"],
                     "J_ring": r["J_ring"], "resid": r["resid"], "step": r["step_center"],
                     "J_ring_k3": best[3]["J_ring"], "J_ring_k4": best[4]["J_ring"],
                     "observer_test": obs, "powers_at_display_center": P.tolist()}
        print(f"{name:12s} k={k} {r['set']}  resid {r['resid']:.3f} step {r['step_center']:.3f} "
              f"J {r['J_ring']:.3f} (k3 {best[3]['J_ring']:.3f}, k4 {best[4]['J_ring']:.3f})  "
              f"observers {obs['mean']:.2f}/{obs['worst']:.2f}", flush=True)
    (OUT / "resolution_sets.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
