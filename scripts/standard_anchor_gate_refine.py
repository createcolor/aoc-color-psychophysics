"""The observer-selected configuration per center: ring gate and adopted drive.

Per center, walk down the training ranking of standard_anchor_selection.py
until a set renders the center and its dE00 = 3 ring through the device path
(conv._feasible_for_targets, tolerance 1.0 dE00). On that set's coefficient
family, sample N_TAU members, render each with powers rounded to 0.1 and adopt
the one with the lowest worst-training-observer departure among those with a
CIE 1964 residual <= RESID_GATE. Reported for the adopted member and for the
solver's member: held-out mean and worst, residual and minimal step.

Usage:
    PYTHONPATH=.:scripts python scripts/standard_anchor_gate_refine.py
Writes output/standard_anchor_gate_refine.json.
"""
from __future__ import annotations

import json

import numpy as np

from common import (CAL, CENTERS, N_TAU, OUT, RING_DE, Population, converter, freqs_of,
                    null_family, ring, step_from_powers)

RESID_GATE = 0.12


def main() -> None:
    conv = converter()
    pop = Population(conv)
    sc = pop.sc
    out = {}
    for name in CENTERS:
        sel = json.loads((OUT / "standard_anchor_selection" /
                          f"{name}.json").read_text())
        t = pop.center(name)
        targets = [t["xyY"]] + ring(conv, t["xyY"], RING_DE)
        for rank, r in enumerate(sel["rows"], 1):
            wls = sorted((float(w) for w in r["set"]), reverse=True)
            freqs = freqs_of(conv, wls)
            if conv._feasible_for_targets(freqs, targets, tol_dE=1.0):
                break
        else:
            print(f"--- {name}: no set passes the ring gate", flush=True)
            continue
        channels = conv._channel_assignment(len(wls))
        c0 = np.asarray(sc.waves_n_xyY_to_ao_intesities(wls, t["xyY"], channels=channels),
                        dtype=float)
        nullv, lo, hi = null_family(sc, wls, c0, channels, pop.ref_cmf)

        def render(inten):
            P = np.round(np.asarray(conv._get_power(freqs, inten, channels=channels), float), 1)
            stim = np.asarray(conv.P_f_to_sd(P, freqs, channels=channels).values, dtype=float)
            return P, pop.de(stim, t), pop.resid(stim, t)

        members = []
        for tau in np.linspace(lo, hi, N_TAU):
            inten = c0 + tau * nullv
            if np.any(inten <= 0):
                continue
            try:
                P, de, resid = render(inten)
            except Exception:                       # noqa: BLE001  (not renderable)
                continue
            members.append({"tau": float(tau), "P": P, "de": de, "resid": resid})
        gated = [m for m in members if m["resid"] <= RESID_GATE] or members
        b = min(gated, key=lambda m: (float(m["de"][pop.train].max()),
                                      float(m["de"][pop.train].mean())))
        P_s, de_s, resid_s = render(c0)

        def report(P, de, resid):
            return {"powers": [float(p) for p in P], "resid": resid,
                    "step": step_from_powers(conv, P, freqs, channels),
                    **{f"test_{k}": v for k, v in pop.stats(de).items()}}

        out[name] = {"set": r["set"], "rank": rank,
                     "tau_frac": (b["tau"] - lo) / (hi - lo) if hi > lo else 0.5,
                     "solver": report(P_s, de_s, resid_s),
                     "adopted": report(b["P"], b["de"], b["resid"])}
        a, s = out[name]["adopted"], out[name]["solver"]
        print(f"--- {name}: rank-{rank} set {tuple(r['set'])} passes the ring gate; adopted member "
              f"{a['test_mean']:.2f}/{a['test_worst']:.2f} resid {a['resid']:.2f} step {a['step']:.2f}"
              f" (solver member {s['test_mean']:.2f}/{s['test_worst']:.2f})", flush=True)
    (OUT / "standard_anchor_gate_refine.json").write_text(json.dumps(
        {"calibration": CAL.name, "ring_de": RING_DE, "resid_gate": RESID_GATE, "n_tau": N_TAU,
         "results": out}, indent=1))


if __name__ == "__main__":
    main()
