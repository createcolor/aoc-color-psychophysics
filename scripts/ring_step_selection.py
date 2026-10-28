"""Primaries for color matching: the resolution criterion (paper Eq. 3), scored
over the target and its neighborhood ring, for every primary set at a center.

At the nominal CIE color center, every primary set at >= 20 nm separation is
solved through the device path (xyY_to_P_f with the set's frequencies, the
member the instrument would command), its powers rounded to the 0.1 drive
quantum, and kept if the rendered color is within 1.0 dE00 of the target.

  gate : the center and the 8 ring colors at dE00 = 3 all render this way;
  score: J = q + the largest single-quantum step over the 9 points, where q is
         the placement error at the center (half a 12-bit step on every channel,
         all sign combinations) and the step is taken against each point's own
         rendered color.

k=3 sets are all ring-scored; k=4 sets are ranked by J at the center and the
ring is evaluated for the --shortlist best (default 200).

Usage:
    PYTHONPATH=.:scripts python scripts/ring_step_selection.py --center blue --primaries 4
Writes output/ring_step_selection/<center>_k<k>.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations

import numpy as np
from numpy.linalg import LinAlgError

# common before ao: it silences colour's matplotlib warning
from common import CAL, CENTERS, MIN_SEPARATION_NM, NOMINAL, OUT, RING_DE, converter, primaries_of, ring
from ao.spectral_converter import XYZ_to_xyY, xyY_to_XYZ, TableBounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=CENTERS)
    ap.add_argument("--primaries", type=int, choices=(3, 4), required=True)
    ap.add_argument("--shortlist", type=int, default=200,
                    help="k=4: ring-score only the N best by center J")
    args = ap.parse_args()
    conv = converter()
    sc = conv._spectral_converter
    k = args.primaries
    ch = conv._channel_assignment(k)
    t = NOMINAL[args.center]
    t_lab = conv._xyz_to_lab(xyY_to_XYZ(t))
    pts = ring(conv, t, RING_DE)

    def solve(freqs, point):
        """Device-path powers (descending wavelength), rounded, or None."""
        try:
            P, _ = conv.xyY_to_P_f(list(point), frequencies=[float(x) for x in freqs])
            P = np.round(np.asarray([P[c - 1] for c in ch], float), 1)
            rep = XYZ_to_xyY(conv.P_f_to_XYZ(P, freqs, channels=ch))
            if conv._ciede2000(conv._xyz_to_lab(xyY_to_XYZ(list(point))),
                               conv._xyz_to_lab(xyY_to_XYZ(rep))) > 1.0:
                return None
            return P
        except (ValueError, TableBounds, LinAlgError, IndexError):
            return None

    def step(freqs, P):
        own = conv._xyz_to_lab(xyY_to_XYZ(XYZ_to_xyY(conv.P_f_to_XYZ(P, freqs, channels=ch))))
        s = 0.0
        for i in range(k):
            for sgn in (0.1, -0.1):
                q = P.copy()
                q[i] += sgn
                if q[i] < 0:
                    continue
                try:
                    s = max(s, conv._ciede2000(own, conv._xyz_to_lab(xyY_to_XYZ(
                        XYZ_to_xyY(conv.P_f_to_XYZ(q, freqs, channels=ch))))))
                except (TableBounds, ValueError):
                    pass
        return s

    def q_term(wl, freqs, P):
        inten = conv._get_intensity(freqs, P, channels=ch)
        prims = [XYZ_to_xyY(np.asarray(sc.integrate_sd_to_XYZ(sc._table_sd(float(w), float(I), c)), float))
                 for w, I, c in zip(wl, inten, ch)]
        xt, yt, Yt = t
        den = 2046.0
        scale = (yt / Yt) / den
        dx = scale * sum(p[2] / p[1] * abs(p[0] - xt) for p in prims)
        dy = scale * sum(p[2] / p[1] * abs(p[1] - yt) for p in prims)
        dY = sum(p[2] for p in prims) / den
        return max(conv._ciede2000(t_lab, conv._xyz_to_lab(xyY_to_XYZ(
            [xt + sx * dx, yt + sy * dy, Yt + sl * dY])))
            for sx in (-1, 1) for sy in (-1, 1) for sl in (-1, 1))

    rows = []
    for n, combo in enumerate(combinations(primaries_of(sc), k), 1):
        if n % 500 == 0:
            print(f"[{datetime.now():%H:%M:%S}] {args.center} k={k}: {n} combos, "
                  f"{len(rows)} renderable", flush=True)
        if min(np.diff(combo)) < MIN_SEPARATION_NM:
            continue
        wl = [float(w) for w in sorted(combo, reverse=True)]
        freqs = conv._snap_frequencies_to_calibration_grid(conv._get_frequency(np.asarray(wl)))
        P = solve(freqs, t)
        if P is None:
            continue
        xyY_r = XYZ_to_xyY(conv.P_f_to_XYZ(P, freqs, channels=ch))
        resid = conv._ciede2000(t_lab, conv._xyz_to_lab(xyY_to_XYZ([xyY_r[0], xyY_r[1], t[2]])))
        s, q = step(freqs, P), q_term(wl, freqs, P)
        rows.append({"set": [int(w) for w in wl], "freqs": [float(f) for f in freqs],
                     "powers": P.tolist(), "resid": resid, "q": q,
                     "step_center": s, "J_center": q + s})
    rows.sort(key=lambda r: r["J_center"])
    print(f"[{datetime.now():%H:%M:%S}] {args.center} k={k}: {len(rows)} renderable at the center",
          flush=True)

    for r in rows[:args.shortlist] if k == 4 else rows:
        freqs = np.asarray(r["freqs"])
        steps = [None if P is None else step(freqs, P) for P in (solve(freqs, p) for p in pts)]
        if all(s is not None for s in steps):
            r["ring_mean_step"] = float(np.mean(steps))
            r["ring_max_step"] = max(steps + [r["step_center"]])
            r["J_ring"] = r["q"] + r["ring_max_step"]

    out = OUT / "ring_step_selection" / f"{args.center}_k{k}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"center": args.center, "target_xyY": t, "primaries": k,
                               "channels": ch, "calibration": CAL.name, "ring_de": RING_DE,
                               "shortlist": args.shortlist if k == 4 else None,
                               "rows": rows}, indent=1))
    best = min((r for r in rows if "J_ring" in r), key=lambda r: r["J_ring"], default=None)
    if best:
        print(f"--- {args.center} k={k}: best {best['set']} P {best['powers']} J {best['J_ring']:.3f} "
              f"(step {best['step_center']:.3f}, resid {best['resid']:.3f})", flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
